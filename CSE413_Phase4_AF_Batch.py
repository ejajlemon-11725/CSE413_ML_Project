#!/usr/bin/env python3
"""Incremental AlphaFold structure download with exact RefSeq sequence/residue verification."""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

ROOT = Path(r'D:\CSE413_ML_Project')
AA3 = {
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}

def pdb_sequence_matches(path, sequence):
    try:
        model = next(PDBParser(QUIET=True).get_structure('af', str(path)).get_models())
        chain = next(model.get_chains())
        residues = [r for r in chain.get_residues() if r.id[0] == ' ']
        if len(residues) != len(sequence):
            return False, f'length mismatch {len(residues)} != {len(sequence)}'
        for position, (res, reference_aa) in enumerate(zip(residues, sequence), 1):
            # AFStore uses zero-based iloc[pos-1], so residue numbering must be 1..L.
            if res.id[1] != position or res.id[2] != ' ':
                return False, f'numbering mismatch at {position}: {res.id}'
            aa = seq1(res.resname, custom_map=AA3, undef_code='X')
            if aa != reference_aa:
                return False, f'sequence mismatch at {position}: {aa} != {reference_aa}'
        return True, 'verified'
    except Exception as e:
        return False, f'PDB parse error: {type(e).__name__}: {e}'

def get_with_retry(session, url, *, stream=False, timeout=45):
    for attempt in range(4):
        try:
            response = session.get(url, timeout=timeout, stream=stream)
            if response.status_code in (429, 500, 502, 503, 504):
                response.close()
                time.sleep(min(30, 2 ** attempt * 3))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 2)
    raise RuntimeError(f'No response after retries: {url}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--limit', type=int, default=25, help='Maximum unique RefSeq proteins to attempt, 0=all')
    ap.add_argument('--timeout', type=int, default=45)
    args = ap.parse_args()
    root = args.root
    dataset = pd.read_csv(root / 'data' / 'Clinical_substitutions.csv', usecols=['protein_id', 'protein_sequence'])
    seqs = dataset.drop_duplicates('protein_id').set_index('protein_id')['protein_sequence'].to_dict()
    if dataset.groupby('protein_id')['protein_sequence'].nunique().gt(1).any():
        raise ValueError('Conflicting reference sequences for one protein ID')
    mapping_path = root / 'results' / 'phase4_mapping_pilot' / 'mapping_pilot.csv'
    mapping = pd.read_csv(mapping_path).fillna('')
    expected = {'refseq', 'uniprot', 'match', 'alphaFold_api'}
    if not expected.issubset(mapping.columns):
        raise ValueError(f'Missing mapping columns: {expected - set(mapping.columns)}')
    eligible = mapping[(mapping['match'] == 'EXACT') & (mapping['alphaFold_api'] == 'AVAILABLE') & (mapping['refseq'].isin(seqs))].copy()
    # Prefer conventional Swiss-Prot-like identifiers, but do NOT assume reviewed status.
    eligible['priority'] = eligible['uniprot'].apply(lambda x: 0 if re.fullmatch(r'[OPQ][0-9][A-Z0-9]{3}[0-9]', str(x)) else 1)
    eligible = eligible.sort_values(['refseq', 'priority', 'uniprot']).drop_duplicates(['refseq', 'uniprot'])
    grouped = eligible.groupby('refseq')['uniprot'].apply(list).to_dict()
    print(f'Unique dataset proteins: {len(seqs)}', flush=True)
    print(f'Unique proteins with exact+available mappings: {len(grouped)}', flush=True)
    print(f'Not mapped/available: {len(set(seqs)-set(grouped))}', flush=True)
    out_dir = root / 'data' / 'alphafold'
    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir = root / 'results' / 'phase4_alphafold_batch'
    result_dir.mkdir(parents=True, exist_ok=True)
    report_path = result_dir / 'download_report.csv'
    existing = set()
    if report_path.exists():
        old = pd.read_csv(report_path)
        if 'refseq' in old:
            existing = set(old.loc[old['status'] == 'VERIFIED_PDB', 'refseq'])
    # deterministic by dataset order; every run resumes successful proteins
    todo = [pid for pid in seqs if pid in grouped]
    if args.limit > 0:
        todo = todo[:args.limit]
    columns = ['refseq', 'uniprot', 'status', 'detail']
    if not report_path.exists():
        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, columns).writeheader()
    s = requests.Session()
    s.headers.update({'User-Agent': 'CSE413-academic-feature-pipeline/1.0'})
    verified, failed, skipped = 0, 0, 0
    for count, pid in enumerate(todo, 1):
        dest = out_dir / f'{pid}.pdb'
        af_copy = out_dir / f'AF-{pid}-F1-model_v4.pdb'
        if dest.exists():
            ok, detail = pdb_sequence_matches(dest, seqs[pid])
            if ok:
                if not af_copy.exists():
                    import shutil
                    shutil.copy2(dest, af_copy)
                print(f'[{count}/{len(todo)}] {pid}: VERIFIED_EXISTING', flush=True)
                skipped += 1
                continue
            print(f'[{count}/{len(todo)}] {pid}: existing PDB invalid ({detail}), attempting replacement', flush=True)
        status, detail, selected = 'FAILED', 'all candidate accessions exhausted', ''
        for acc in grouped[pid]:
            selected = acc
            try:
                meta = get_with_retry(s, f'https://alphafold.ebi.ac.uk/api/prediction/{acc}', timeout=args.timeout).json()
                entries = meta if isinstance(meta, list) else [meta]
                urls = [x.get('pdbUrl') for x in entries if isinstance(x, dict) and x.get('pdbUrl')]
                if not urls:
                    detail = f'no PDB URL for {acc}'
                    continue
                url = urls[0]
                temp = dest.with_suffix('.pdb.part')
                try:
                    response = get_with_retry(s, url, stream=True, timeout=args.timeout)
                    with response, open(temp, 'wb') as f:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    ok, check = pdb_sequence_matches(temp, seqs[pid])
                    if not ok:
                        detail = f'{acc}: {check}'
                        continue
                    temp.replace(dest)
                    import shutil
                    shutil.copy2(dest, af_copy)
                    status, detail = 'VERIFIED_PDB', f'{len(seqs[pid])} residues exact'
                    break
                finally:
                    temp.unlink(missing_ok=True)
            except (requests.RequestException, ValueError, KeyError, OSError) as e:
                detail = f'{acc}: {type(e).__name__}: {str(e)[:130]}'
        with open(report_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, columns).writerow(dict(refseq=pid, uniprot=selected, status=status, detail=detail))
        if status == 'VERIFIED_PDB':
            verified += 1
        else:
            failed += 1
        print(f'[{count}/{len(todo)}] {pid} -> {selected}: {status} ({detail})', flush=True)
    print(f'SUMMARY: downloaded_verified={verified}; already_verified={skipped}; failed={failed}; attempted={len(todo)}', flush=True)
    print(f'REPORT: {report_path}', flush=True)
    print('Models and sealed test labels were not accessed.', flush=True)

if __name__ == '__main__':
    main()
