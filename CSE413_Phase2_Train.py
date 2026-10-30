#!/usr/bin/env python3
"""Train independent CSE413 ML models on biochemical + cached ESM2-t6 features.

Uses exactly the frozen split from Phase 1. Fits on TRAIN only, evaluates VALIDATION only.
Never fits, predicts, scores or examines labels for the sealed TEST partition.

Windows: python -u CSE413_Phase2_Train.py
"""
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import CSE413_Phase1_Baseline as base

ROOT = Path(r'D:\CSE413_ML_Project')
CACHE = ROOT / 'data' / 'esm2_t6_features'
OUT = ROOT / 'results' / 'phase2_biochemical_plus_esm2'
EXPECTED_ROWS = 62727
EXPECTED_ESM_DIMS = 322


def load_esm(expected_hash, n):
    config_path = CACHE / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing ESM cache manifest: {config_path}')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if (config.get('source_sha256') != expected_hash.lower()
            or config.get('source_rows') != n
            or config.get('model') != 'facebook/esm2_t6_8M_UR50D'
            or config.get('window') != 255
            or len(config.get('feature_columns', [])) != EXPECTED_ESM_DIMS):
        raise ValueError('ESM cache metadata differs from the Phase 1 CSV / expected model or window. STOP.')

    shards = sorted(CACHE.glob('rows_*.npz'))
    if not shards:
        raise FileNotFoundError(f'No ESM shards in {CACHE}')
    X = np.empty((n, EXPECTED_ESM_DIMS), dtype=np.float32)
    next_start = 0
    for i, path in enumerate(shards, 1):
        with np.load(path, allow_pickle=False) as z:
            row_index = z['row_index']
            chunk = z['X']
            row_hash = str(z['csv_sha256'].item())
        end = next_start + len(row_index)
        if (end > n or not np.array_equal(row_index, np.arange(next_start, end))
                or chunk.shape != (len(row_index), EXPECTED_ESM_DIMS)
                or row_hash.lower() != expected_hash.lower()
                or not np.isfinite(chunk).all()
                or path.name != f'rows_{next_start:06d}_{end:06d}.npz'):
            raise ValueError(f'ESM shard mismatch, missing or unordered rows at: {path.name}. STOP.')
        X[next_start:end] = chunk
        next_start = end
        if i % 100 == 0 or end == n:
            print(f'  Verified ESM rows: {end}/{n}', flush=True)
    if next_start != n:
        raise ValueError(f'ESM cache incomplete: covered {next_start}/{n} rows. STOP.')
    return pd.DataFrame(X, columns=config['feature_columns'])


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = base.SRC
    split_path = base.SPLIT
    if not csv_path.is_file() or not split_path.is_file():
        raise FileNotFoundError('Missing Phase 1 CSV or frozen split; stop rather than create a new split.')
    data_hash = base.sha(csv_path)
    with np.load(split_path, allow_pickle=False) as z:
        train = z['train'].astype(int)
        val = z['val'].astype(int)
        test = z['test'].astype(int)
        split_csv_hash = str(z['csv_sha256'].item())
    if data_hash != split_csv_hash:
        raise ValueError('Frozen split checksum differs from source CSV. STOP.')

    print('Loading Phase 1 dataset and frozen split...', flush=True)
    df = pd.read_csv(csv_path, usecols=['protein_id','protein_sequence','mutant','annotation'], low_memory=False)
    if len(df) != EXPECTED_ROWS or not df.annotation.isin(['Pathogenic','Benign']).all():
        raise ValueError('Dataset row count / labels differ from Phase 1; STOP.')
    if df[['protein_id','protein_sequence','mutant','annotation']].isna().any().any():
        raise ValueError('Missing values in source dataset; STOP.')
    if df.duplicated(['protein_id','mutant']).any():
        raise ValueError('Duplicate protein + mutation detected; STOP.')
    n = len(df)
    if (len(set(np.concatenate([train,val,test]).tolist())) != n
            or sum(map(len,(train,val,test))) != n
            or any((idx.min() < 0 or idx.max() >= n) for idx in (train,val,test))):
        raise ValueError('Frozen split row coverage is invalid. STOP.')
    proteins = df.protein_id.astype(str).to_numpy()
    if (set(proteins[train]) & set(proteins[val])
            or set(proteins[train]) & set(proteins[test])
            or set(proteins[val]) & set(proteins[test])):
        raise ValueError('Protein-ID leakage in frozen split. STOP.')
    print(f'Frozen split: train={len(train)} validation={len(val)} SEALED test={len(test)}', flush=True)

    print('Validating all cached ESM features...', flush=True)
    X_esm = load_esm(data_hash, n)
    print('Generating biochemical features...', flush=True)
    X_bio = base.build_features(df).reset_index(drop=True)
    if len(X_bio) != n or not np.isfinite(X_bio.to_numpy(dtype=np.float32)).all():
        raise ValueError('Biochemical features invalid. STOP.')
    X = pd.concat([X_bio, X_esm], axis=1)
    if X.columns.duplicated().any() or len(X) != n:
        raise ValueError('Feature name/row alignment error. STOP.')
    del X_bio, X_esm
    y = df.annotation.eq('Pathogenic').astype(np.int8).to_numpy()
    print(f'Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns', flush=True)
    print('Training only on frozen train; evaluation only on validation.', flush=True)

    models = {
        'Logistic_Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1800, random_state=base.SEED)),
        'Random_Forest': RandomForestClassifier(n_estimators=220, max_depth=18, min_samples_leaf=3,
                                                n_jobs=8, random_state=base.SEED),
        'XGBoost': XGBClassifier(n_estimators=350, max_depth=5, learning_rate=.05, subsample=.85,
                                 colsample_bytree=.9, objective='binary:logistic', eval_metric='logloss',
                                 tree_method='hist', n_jobs=8, random_state=base.SEED),
    }
    results = []
    for name, model in models.items():
        print(f'Training {name}...', flush=True)
        t0 = time.time()
        model.fit(X.iloc[train], y[train])
        prob = model.predict_proba(X.iloc[val])[:, 1]
        m = base.metrics(y[val], prob)
        record = {'model': name, 'partition': 'validation', 'n': len(val), 'threshold': 0.5,
                  'training_seconds': round(time.time()-t0, 2), **m}
        results.append(record)
        joblib.dump(model, OUT / f'{name}.joblib')
        pd.DataFrame({'row_index':val, 'protein_id':proteins[val], 'y_true':y[val],
                      'probability':prob}).to_csv(OUT / f'{name}_validation_predictions.csv', index=False)
        print(f'  VAL AUC={m["roc_auc"]:.5f} ACC={m["accuracy"]:.5f} F1={m["f1"]:.5f} '
              f'AP={m["average_precision"]:.5f} ({record["training_seconds"]:.1f}s)', flush=True)

    current = pd.DataFrame(results)
    current.to_csv(OUT / 'validation_metrics.csv', index=False)
    prior_path = base.OUT / 'validation_metrics.csv'
    if prior_path.exists():
        previous = pd.read_csv(prior_path)
        combined = current.merge(previous[['model','accuracy','roc_auc','f1']], on='model',
                                 how='left', suffixes=('_phase2','_phase1'))
        for metric in ('accuracy','roc_auc','f1'):
            combined[f'{metric}_change'] = combined[f'{metric}_phase2'] - combined[f'{metric}_phase1']
        combined.to_csv(OUT / 'vs_phase1_validation.csv', index=False)
        print('\nCompared against Phase 1 VALIDATION (percentage-point changes):', flush=True)
        for row in combined.itertuples():
            print(f'  {row.model}: accuracy change={getattr(row,"accuracy_change")*100:+.2f} pp; '
                  f'AUC change={getattr(row,"roc_auc_change")*100:+.2f} pp',flush=True)
    base.write_json(OUT / 'manifest.json', {
        'project':'CSE413 Machine Learning Theory — Phase 2',
        'csv_sha256':data_hash, 'frozen_split_path':str(split_path),
        'frozen_split_sha256':base.sha(split_path),
        'esm_cache':str(CACHE), 'esm_model':'facebook/esm2_t6_8M_UR50D',
        'esm_feature_type':'320 mutation-site embedding deltas + L2 magnitude + WT/mutant cosine, 255-aa window',
        'total_features':len(X.columns), 'feature_names':list(X.columns),
        'train_n':len(train),'validation_n':len(val),'sealed_test_n':len(test),
        'models':list(models),'seed':base.SEED,
        'caveats':['The ESM encoder is pretrained; only classical ML classifiers were fitted.',
                   'Exact protein-ID disjointness is not sequence-homology disjointness.',
                   'Final test remains sealed; validation results are not unbiased final-test performance.']
    })
    print(f'\nDONE: {OUT}', flush=True)
    print(f'Elapsed: {time.time()-started:.1f} seconds', flush=True)
    print('FINAL TEST REMAINS SEALED.',flush=True)


if __name__ == '__main__':
    main()
