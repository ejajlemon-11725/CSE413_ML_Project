#!/usr/bin/env python3
"""Download sequence-verified AlphaFold PDBs for Phase-4 RefSeq pilot.

Uses exact mappings from mapping_pilot.csv, prefers unique reviewed UniProt entry,
validates AlphaFold PDB residue identifiers and complete amino-acid sequence.
No training, labels, or split access.
"""
import argparse
import io
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = 'https://rest.uniprot.org'


def new_session():
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.headers.update({'User-Agent': 'CSE413-AlphaFold-verified-pilot/1.0'})
    return s


def get_json(s, url):
    r = s.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def check_pdb(pdb_bytes, expected):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('af', io.StringIO(pdb_bytes.decode('utf-8')))
    models = list(structure.get_models())
    if len(models) != 1:
        return False, 'MODEL_COUNT_MISMATCH'
    chains = list(models[0].get_chains())
    if len(chains) != 1:
        return False, 'CHAIN_COUNT_MISMATCH'
    found = {}
    for residue in chains[0].get_residues():
        flag, index, insertion = residue.id
        if flag != ' ' or insertion != ' ':
            return False, 'NONSTANDARD_RESIDUE_NUMBERING'
        if index in found:
            return False, 'DUPLICATE_RESIDUE_INDEX'
        if 'CA' not in residue:
            return False, f'MISSING_CA_{index}'
        aa = seq1(residue.resname, custom_map={'UNK': 'X'})
        if len(aa) != 1:
            return False, f'UNKNOWN_RESIDUE_{index}'
        found[index] = aa
    if set(found) != set(range(1, len(expected) + 1)):
        return False, f'RESIDUE_NUMBERING_MISMATCH_pdb={len(found)}_csv={len(expected)}'
    pdb_sequence = ''.join(found[k] for k in range(1, len(expected) + 1))
    if pdb_sequence != expected:
        return False, 'SEQUENCE_MISMATCH'
    return True, 'EXACT_FULL_LENGTH'


def main(args):
    root = Path(args.root)
    mapping_path = root / 'results' / 'phase4_mapping_pilot' / 'mapping_pilot.csv'
    dataset_path = root / 'data' / 'Clinical_substitutions.csv'
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)
    map_df = pd.read_csv(mapping_path, dtype=str).fillna('')
    required = {'refseq', 'uniprot', 'match', 'alphaFold_api'}
    if not required.issubset(map_df.columns):
        raise ValueError(f'Unexpected mapping report columns: {list(map_df.columns)}')
    proteins = pd.read_csv(dataset_path, usecols=['protein_id', 'protein_sequence']).drop_duplicates('protein_id')
    sequence = dict(zip(proteins.protein_id.astype(str), proteins.protein_sequence.astype(str)))
    candidates = map_df[(map_df['match'] == 'EXACT') & (map_df['alphaFold_api'] == 'AVAILABLE')]
    out_dir = root / 'data' / 'alphafold'
    report_dir = root / 'results' / 'phase4_alphafold_pilot'
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    s = new_session()
    records = []
    entries = {}
    ref_list = list(dict.fromkeys(map_df.refseq))[:args.limit]
    for ref in ref_list:
        if ref not in sequence:
            records.append({'refseq': ref, 'status': 'REFSEQ_NOT_IN_DATASET'})
            continue
        expected = sequence[ref].strip().upper()
        sub = candidates[candidates.refseq == ref]
        info = []
        for acc in sorted(set(sub.uniprot)):
            if acc not in entries:
                try:
                    entries[acc] = get_json(s, f'{API}/uniprotkb/{acc}.json')
                except (requests.RequestException, ValueError):
                    entries[acc] = None
            entry = entries[acc]
            if not entry or entry.get('sequence', {}).get('value', '').upper() != expected:
                continue
            is_reviewed = 'reviewed' in entry.get('entryType', '').lower() and 'unreviewed' not in entry.get('entryType', '').lower()
            info.append((acc, is_reviewed))
        reviewed = [acc for acc, flag in info if flag]
        if len(reviewed) == 1:
            chosen, selection = reviewed[0], 'UNIQUE_REVIEWED'
        elif len(reviewed) > 1:
            print(f'{ref}: AMBIGUOUS multiple reviewed accessions: {reviewed}', flush=True)
            records.append({'refseq': ref, 'status': 'AMBIGUOUS_REVIEWED', 'candidates': ','.join(reviewed)})
            continue
        elif len(info) == 1:
            chosen, selection = info[0][0], 'ONLY_EXACT_CANDIDATE'
        else:
            print(f'{ref}: SKIPPED (no unique canonical choice, candidates={info})', flush=True)
            records.append({'refseq': ref, 'status': 'AMBIGUOUS_OR_NO_MATCH', 'candidates': ','.join(a for a, _ in info)})
            continue
        try:
            predictions = get_json(s, f'https://alphafold.ebi.ac.uk/api/prediction/{chosen}')
            if not isinstance(predictions, list) or not predictions:
                raise ValueError('Empty AlphaFold prediction response')
            matching = [p for p in predictions if p.get('pdbUrl') and p.get('uniprotAccession', chosen) == chosen]
            if not matching:
                raise ValueError('No PDB URL for matching accession')
            if len(matching) != 1:
                raise ValueError('Multiple PDB predictions: manual review required')
            pdb_url = matching[0]['pdbUrl']
            if not pdb_url.startswith('https://alphafold.ebi.ac.uk/files/'):
                raise ValueError('Unexpected AlphaFold PDB URL')
            response = s.get(pdb_url, timeout=120)
            response.raise_for_status()
            verified, reason = check_pdb(response.content, expected)
            if not verified:
                print(f'{ref} -> {chosen}: REJECTED {reason}', flush=True)
                records.append({'refseq': ref, 'uniprot': chosen, 'selection': selection, 'status': reason})
                continue
            target = out_dir / f'{ref}.pdb'
            if target.exists() and target.read_bytes() != response.content:
                raise RuntimeError(f'Refusing to overwrite a different existing file: {target}')
            target.write_bytes(response.content)
            records.append({'refseq': ref, 'uniprot': chosen, 'selection': selection, 'status': 'VERIFIED_PDB', 'pdb_path': str(target), 'sequence_length': len(expected)})
            print(f'{ref} -> {chosen}: VERIFIED_PDB {len(expected)} residues, saved {target.name}', flush=True)
        except Exception as e:
            print(f'{ref} -> {chosen}: ERROR {type(e).__name__}: {e}', flush=True)
            records.append({'refseq': ref, 'uniprot': chosen, 'selection': selection, 'status': 'ERROR', 'detail': str(e)[:300]})
        time.sleep(0.2)
    result = pd.DataFrame(records)
    result.to_csv(report_dir / 'download_report.csv', index=False)
    print('\nSTATUS COUNTS:\n' + result.status.value_counts().to_string(), flush=True)
    print('REPORT:', report_dir / 'download_report.csv', flush=True)
    print('No model, cached ESM feature, labels, or frozen split modified.', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'D:\CSE413_ML_Project')
    ap.add_argument('--limit', type=int, default=10)
    main(ap.parse_args())
