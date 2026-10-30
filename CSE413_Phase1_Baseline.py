#!/usr/bin/env python3
"""Independent CSE413 baseline: no prior experiment data, checkpoints, or results.
Usage: python CSE413_Phase1_Baseline.py
Only TRAIN/VALIDATION are evaluated. TEST stays sealed.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(r'D:\CSE413_ML_Project')
SRC = ROOT / 'data' / 'Clinical_substitutions.csv'
OUT = ROOT / 'results' / 'phase1_baseline'
SPLIT = ROOT / 'data' / 'cse413_protein_group_split_seed413.npz'
SEED = 413
AA = 'ACDEFGHIKLMNPQRSTVWY'
# Broad chemistry descriptors; explicit approximate heuristic scales used for baseline.
HYDRO = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-3.5,-1.6,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
VOLUME = dict(zip(AA, [88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.2,114.1,143.8,112.7,173.4,89.,116.1,140.,227.8,193.6]))
CHARGE = {a: (1 if a in 'KR' else -1 if a in 'DE' else 0) for a in AA}
BLOSUM = substitution_matrices.load('BLOSUM62')


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest().upper()


def build_features(df):
    rows=[]; errors=[]
    for ix,(seq, mut) in enumerate(zip(df.protein_sequence, df.mutant)):
        seq=str(seq).upper(); match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',str(mut).upper().strip())
        if not match:
            errors.append((ix,'mutation_format')); continue
        wt,pos,mt=match.group(1),int(match.group(2)),match.group(3)
        if wt not in AA or mt not in AA or pos<1 or pos>len(seq) or seq[pos-1]!=wt or wt==mt:
            errors.append((ix,'reference_or_substitution_invalid')); continue
        window=seq[max(0,pos-8):min(len(seq),pos+7)]
        rows.append(dict(row_index=ix, pos_fraction=pos/len(seq), log_seq_length=np.log1p(len(seq)),
            wt_index=AA.index(wt), mut_index=AA.index(mt),
            blosum62=float(BLOSUM[wt,mt]), delta_hydropathy=HYDRO[mt]-HYDRO[wt],
            abs_delta_hydropathy=abs(HYDRO[mt]-HYDRO[wt]),
            delta_volume=VOLUME[mt]-VOLUME[wt], abs_delta_volume=abs(VOLUME[mt]-VOLUME[wt]),
            delta_charge=CHARGE[mt]-CHARGE[wt], charge_flip=int(CHARGE[mt]!=CHARGE[wt]),
            to_proline=int(mt=='P'), from_proline=int(wt=='P'),
            to_glycine=int(mt=='G'), from_glycine=int(wt=='G'),
            to_cysteine=int(mt=='C'), from_cysteine=int(wt=='C'),
            window_hydropathy=np.mean([HYDRO[a] for a in window if a in HYDRO]) if any(a in HYDRO for a in window) else 0,
            window_charge=sum(CHARGE.get(a,0) for a in window)/len(window)))
    if errors:
        sample=errors[:8]
        raise ValueError(f'Found {len(errors)} invalid variants, first examples: {sample}; do not silently drop or modify labels')
    feat=pd.DataFrame(rows).set_index('row_index')
    # One-hot codes to avoid pretending AA indices are ordinal.
    for field in ('wt_index','mut_index'):
        for k in range(len(AA)):
            feat[f'{field}_{AA[k]}']=(feat[field]==k).astype(int)
        feat=feat.drop(columns=field)
    return feat


def metrics(y,p):
    pred=(p>=.5).astype(int)
    return {k:float(v) for k,v in {
        'accuracy':accuracy_score(y,pred), 'precision':precision_score(y,pred,zero_division=0),
        'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),
        'mcc':matthews_corrcoef(y,pred),'roc_auc':roc_auc_score(y,p),
        'average_precision':average_precision_score(y,p)}.items()}


def main():
    if not SRC.exists():raise FileNotFoundError(SRC)
    OUT.mkdir(parents=True,exist_ok=True)
    data_sha=sha(SRC)
    df=pd.read_csv(SRC, usecols=['protein_id','protein_sequence','mutant','annotation'],low_memory=False)
    if len(df)!=62727:raise ValueError(f'Expected 62727 source rows, got {len(df)}')
    if not df.annotation.isin(['Pathogenic','Benign']).all():raise ValueError('Unexpected labels')
    if df[['protein_id','protein_sequence','mutant','annotation']].isna().any().any():raise ValueError('Missing source values')
    if df.duplicated(['protein_id','mutant']).any():raise ValueError('Duplicate protein + mutation')
    y=df.annotation.eq('Pathogenic').astype(int).to_numpy()
    groups=df.protein_id.astype(str).to_numpy()
    print('Generating biochemical features for',len(df),'variants...',flush=True)
    X=build_features(df)
    if len(X)!=len(df) or X.isna().any().any():raise RuntimeError('Feature alignment / NaN error')
    if SPLIT.exists():
        with np.load(SPLIT,allow_pickle=False) as s:
            tr=s['train'];va=s['val'];te=s['test']; recorded_sha=str(s['csv_sha256'].item())
        if recorded_sha!=data_sha:raise RuntimeError('Saved split CSV checksum differs; STOP')
        print('Reusing existing FROZEN protein-group split',flush=True)
    else:
        all_ids=np.arange(len(df))
        outer=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED)
        dev,te=next(outer.split(all_ids,y,groups))
        inner=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=SEED+1)
        trloc,valoc=next(inner.split(dev,y[dev],groups[dev]))
        tr,va=dev[trloc],dev[valoc]
        SPLIT.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(SPLIT, train=tr, val=va, test=te, csv_sha256=np.array(data_sha))
        print('FROZEN new protein-ID-disjoint split saved',flush=True)
    assert sum(map(len,(tr,va,te)))==len(df)
    assert len(set(np.concatenate([tr,va,te]).tolist()))==len(df)
    assert not (set(groups[tr])&set(groups[va]) or set(groups[tr])&set(groups[te]) or set(groups[va])&set(groups[te]))
    print('Train/Validation/SEALED Test:',len(tr),len(va),len(te),flush=True)
    print('Note: protein-ID grouping is NOT a sequence-homology audit.',flush=True)
    specs={
        'Logistic_Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1200,random_state=SEED)),
        'Random_Forest':RandomForestClassifier(n_estimators=220,max_depth=18,min_samples_leaf=3,n_jobs=-1,random_state=SEED),
        'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=220,max_leaf_nodes=31,random_state=SEED),
        'XGBoost':XGBClassifier(n_estimators=350,max_depth=5,learning_rate=.05,subsample=.85,colsample_bytree=.9,
            objective='binary:logistic',eval_metric='logloss',tree_method='hist',n_jobs=8,random_state=SEED),
    }
    records=[]
    for name,estimator in specs.items():
        print('Training',name,flush=True)
        estimator.fit(X.iloc[tr],y[tr]);prob=estimator.predict_proba(X.iloc[va])[:,1]
        rec={'model':name,'partition':'validation','n':len(va),'threshold':.5,**metrics(y[va],prob)}
        records.append(rec)
        joblib.dump(estimator,OUT/f'{name}.joblib')
        pd.DataFrame({'row_index':va,'y_true':y[va],'probability':prob,'protein_id':groups[va]}).to_csv(OUT/f'{name}_validation_predictions.csv',index=False)
        print(f'  VAL AUC={rec["roc_auc"]:.5f} ACC={rec["accuracy"]:.5f} F1={rec["f1"]:.5f}',flush=True)
    pd.DataFrame(records).to_csv(OUT/'validation_metrics.csv',index=False)
    write_json(OUT/'manifest.json',{'source_csv':str(SRC),'csv_sha256':data_sha,'split_path':str(SPLIT),'split_sha256':sha(SPLIT),
        'seed':SEED,'n_train':len(tr),'n_validation':len(va),'n_sealed_test':len(te),
        'split_method':'GroupShuffleSplit by exact protein_id (not sequence-homology grouping)',
        'features':list(X.columns),'models':list(specs),'threshold':.5,
        'warning':'No final test labels or metrics examined; no prior TriModal project models/features/results reused'})
    print('DONE:',OUT,flush=True)
    print('FINAL TEST REMAINS SEALED. Use validation results to develop models.',flush=True)

if __name__=='__main__':main()
