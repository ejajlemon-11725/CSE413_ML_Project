#!/usr/bin/env python3
"""Pilot: map 10 RefSeq protein accessions to UniProtKB and verify exact sequences.
Read-only on the dataset and split; writes a standalone mapping report.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = 'https://rest.uniprot.org'


def session_factory():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=['GET', 'POST'])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.headers.update({'User-Agent': 'CSE413-educational-mapping-pilot/1.0'})
    return s


def mapping_results(s, ids):
    r = s.post(f'{API}/idmapping/run', data={'from': 'RefSeq_Protein', 'to': 'UniProtKB', 'ids': ','.join(ids)}, timeout=60)
    r.raise_for_status()
    job = r.json().get('jobId')
    if not job:
        raise RuntimeError('UniProt job ID absent: ' + r.text[:400])
    print('UniProt mapping job:', job, flush=True)
    for attempt in range(45):
        status = s.get(f'{API}/idmapping/status/{job}', timeout=45)
        status.raise_for_status()
        state = status.json()
        if state.get('jobStatus') == 'FAILED':
            raise RuntimeError('UniProt mapping job failed: ' + str(state))
        if state.get('jobStatus') in (None, 'FINISHED'):
            break
        time.sleep(2)
    else:
        raise TimeoutError('UniProt mapping job not finished within polling window')
    url = f'{API}/idmapping/results/{job}?size=500'
    found, failed = [], []
    while url:
        result = s.get(url, timeout=60)
        result.raise_for_status()
        body = result.json()
        found.extend(body.get('results', []))
        failed.extend(body.get('failedIds', []))
        url = None
        # UniProt provides pagination via the HTTP Link header.
        link = result.headers.get('Link', '')
        for part in link.split(','):
            if 'rel="next"' in part:
                url = part.split(';')[0].strip().strip('<>')
                break
    return found, failed


def accession(to):
    if isinstance(to, str):
        return to
    if isinstance(to, dict):
        return to.get('primaryAccession') or to.get('uniProtkbId')
    return None


def run(args):
    root = Path(args.root)
    csv = root / 'data' / 'Clinical_substitutions.csv'
    if not csv.is_file():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv, usecols=['protein_id', 'protein_sequence'])
    proteins = df.drop_duplicates('protein_id').head(args.limit)
    ids = proteins['protein_id'].astype(str).tolist()
    seq_by_id = dict(zip(proteins.protein_id.astype(str), proteins.protein_sequence.astype(str)))
    output = root / 'results' / 'phase4_mapping_pilot'
    output.mkdir(parents=True, exist_ok=True)
    print(f'Checking {len(ids)} RefSeq accessions (out of {df.protein_id.nunique()} proteins).', flush=True)
    s = session_factory()
    mapped, failed = mapping_results(s, ids)
    pairs = {}
    for item in mapped:
        ref, acc = item.get('from'), accession(item.get('to'))
        if ref in seq_by_id and acc:
            pairs.setdefault(ref, set()).add(acc)
    records = []
    cache = {}
    for ref in ids:
        expected = seq_by_id[ref].strip().upper()
        candidates = sorted(pairs.get(ref, []))
        if not candidates:
            records.append({'refseq': ref, 'uniprot': '', 'match': 'NO_MAPPING', 'ref_length': len(expected), 'uniprot_length': '', 'alphaFold_api': 'NOT_CHECKED'})
            print(f'{ref}: NO_MAPPING', flush=True)
            continue
        for acc in candidates:
            if acc not in cache:
                resp = s.get(f'{API}/uniprotkb/{acc}.json', timeout=60)
                if resp.status_code != 200:
                    cache[acc] = None
                else:
                    cache[acc] = resp.json()
            entry = cache[acc]
            got = (entry or {}).get('sequence', {}).get('value', '').upper()
            exact = bool(got) and got == expected
            label = 'EXACT' if exact else ('SEQUENCE_MISMATCH' if got else 'UNIPROT_NOT_FOUND')
            af_status = 'NOT_CHECKED'
            if exact:
                try:
                    ar = s.get(f'https://alphafold.ebi.ac.uk/api/prediction/{acc}', timeout=40)
                    if ar.status_code == 200 and ar.json():
                        af_status = 'AVAILABLE'
                    else:
                        af_status = f'HTTP_{ar.status_code}'
                except (requests.RequestException, ValueError):
                    af_status = 'LOOKUP_ERROR'
            records.append({'refseq': ref, 'uniprot': acc, 'match': label, 'ref_length': len(expected), 'uniprot_length': len(got) if got else '', 'alphaFold_api': af_status})
            print(f'{ref} -> {acc}: {label}; AlphaFold={af_status}', flush=True)
    report = pd.DataFrame.from_records(records)
    path = output / 'mapping_pilot.csv'
    report.to_csv(path, index=False)
    summary = {'refseq_checked': len(ids), 'mapping_pairs': len(mapped), 'failedIds': failed, 'exact_pairs': int((report['match'] == 'EXACT').sum()), 'alphafold_available_exact_pairs': int(((report['match'] == 'EXACT') & (report['alphaFold_api'] == 'AVAILABLE')).sum())}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nSUMMARY:', json.dumps(summary), flush=True)
    print('SAVED:', path, flush=True)
    print('No dataset rows, cached ESM features, models, or frozen split were changed.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=r'D:\CSE413_ML_Project')
    p.add_argument('--limit', type=int, default=10)
    run(p.parse_args())
