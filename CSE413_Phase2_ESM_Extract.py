#!/usr/bin/env python3
"""CSE413 independent, label-free ESM2 mutation-window delta feature extraction.

Windows usage:
  python -u CSE413_Phase2_ESM_Extract.py --limit 100
  python -u CSE413_Phase2_ESM_Extract.py

Output consists of deterministic row-aligned .npz shards, with validation metadata.
Does not train/evaluate, access labels, or modify the frozen dataset split.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, EsmModel

ROOT = Path(r'D:\CSE413_ML_Project')
CSV = ROOT / 'data' / 'Clinical_substitutions.csv'
# Updated output folder to avoid conflict with the older 8M features
OUT = ROOT / 'data' / 'esm2_t33_650M_features'
# Updated model to 650M
MODEL = 'facebook/esm2_t33_650M_UR50D'
AAS = set('ACDEFGHIKLMNPQRSTVWY')
MUT = re.compile(r'([A-Z])(\d+)([A-Z])\Z')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build_windows(df, offset, window_length):
    refs, alts, positions = [], [], []
    left = window_length // 2
    for j, (seq, mutation) in enumerate(zip(df['protein_sequence'], df['mutant'])):
        seq = str(seq).upper()
        m = MUT.fullmatch(str(mutation).upper().strip())
        if not m:
            raise ValueError(f'Row {offset+j}: invalid mutation notation {mutation!r}')
        wt, site, alt = m[1], int(m[2])-1, m[3]
        if not seq or site < 0 or site >= len(seq) or seq[site] != wt or wt not in AAS or alt not in AAS or wt == alt:
            raise ValueError(f'Row {offset+j}: reference/mutant mismatch {mutation!r}')
        start = max(0, site-left)
        end = min(len(seq), start + window_length)
        start = max(0, end-window_length)
        local = site-start
        ref = seq[start:end]
        altseq = ref[:local] + alt + ref[local+1:]
        refs.append(ref)
        alts.append(altseq)
        positions.append(local)
    return refs, alts, positions


def extract_chunk(df, row_start, tokenizer, model, device, window_length, batch_size):
    ref_windows, alt_windows, pos = build_windows(df, row_start, window_length)
    feats=[]
    for a in range(0, len(df), batch_size):
        b = min(a+batch_size, len(df))
        texts = ref_windows[a:b] + alt_windows[a:b]
        tokens = tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
        tokens = {k:v.to(device) for k,v in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
        n=b-a
        positions=torch.tensor(pos[a:b], dtype=torch.long, device=device)+1  # ESM BOS
        ids=torch.arange(n, device=device)
        wt=hidden[ids, positions].float()
        mut=hidden[ids+n, positions].float()
        delta=mut-wt
        length=torch.linalg.vector_norm(delta, dim=1)
        cosine=torch.nn.functional.cosine_similarity(wt,mut,dim=1)
        result=torch.cat([delta, length[:,None], cosine[:,None]],dim=1)
        feats.append(result.cpu().numpy().astype(np.float32))
    return np.concatenate(feats,axis=0)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--limit',type=int,default=0,help='0 means all rows; first use 100 as a pilot')
    # Reduced default batch-size for 650M model to avoid Out-Of-Memory (OOM) errors on GPU
    ap.add_argument('--batch-size',type=int,default=4,help='number of mutations per GPU forward pass')
    ap.add_argument('--shard-size',type=int,default=100)
    ap.add_argument('--window',type=int,default=255)
    args=ap.parse_args()
    if not (1<=args.window<=900):ap.error('--window must be 1..900')
    if args.batch_size<=0 or args.shard_size<=0 or args.limit<0:ap.error('batch-size, shard-size positive; limit nonnegative')
    if not CSV.is_file():raise FileNotFoundError(CSV)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this run; no GPU visible')
    csv_hash=sha256(CSV)
    df=pd.read_csv(CSV, usecols=['protein_sequence','mutant'], low_memory=False)
    total=len(df)
    count=min(total,args.limit) if args.limit else total
    print(f'CSV rows={total}, selected={count}, SHA256={csv_hash[:16]}...',flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    
    # 650M model has 1280 dimensions, so generating 1280 labels
    config={'source_sha256':csv_hash,'source_rows':total,'model':MODEL,'window':args.window,'feature_columns':
            [f'esm_delta_{j:03d}' for j in range(1280)]+['esm_delta_l2','esm_wt_mut_cosine']}
    config_file=OUT/'config.json'
    if config_file.exists():
        previous=json.loads(config_file.read_text(encoding='utf-8'))
        if previous!=config:raise RuntimeError('ESM cache config differs from current CSV/model/window. STOP; use a distinct output folder.')
    else:
        config_file.write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Loading ESM2 GPU encoder...',flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=EsmModel.from_pretrained(MODEL).eval().to('cuda')
    done=0
    for start in range(0,count,args.shard_size):
        end=min(start+args.shard_size,total)
        target=OUT/f'rows_{start:06d}_{end:06d}.npz'
        if target.exists():
            with np.load(target,allow_pickle=False) as z:
                # 1280 delta + 1 L2 + 1 Cosine = 1282 total features
                valid=(np.array_equal(z['row_index'],np.arange(start,end)) and
                       z['X'].shape==(end-start,1282) and
                       np.isfinite(z['X']).all() and
                       str(z['csv_sha256'].item())==csv_hash)
            if not valid:raise RuntimeError(f'Existing shard invalid; STOP: {target}')
            print(f'REUSE {start}-{end-1}',flush=True)
        else:
            features=extract_chunk(df.iloc[start:end],start,tok,model,'cuda',args.window,args.batch_size)
            if features.shape!=(end-start,1282) or not np.isfinite(features).all():
                raise RuntimeError(f'Feature shape / NaN at {start}-{end}')
            temp=target.with_suffix('.partial.npz')
            np.savez_compressed(temp,row_index=np.arange(start,end),X=features,csv_sha256=np.array(csv_hash))
            temp.replace(target)
            print(f'SAVED {start}-{end-1} -> {target.name}',flush=True)
        done=end
    print(f'DONE {done}/{total} rows. Cache: {OUT}',flush=True)
    print('No labels or final-test results were examined. Frozen split unchanged.',flush=True)

if __name__=='__main__':main()