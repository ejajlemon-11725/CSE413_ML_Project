#!/usr/bin/env python3
"""CSE413 independent Phase 3: grouped CV XGBoost tuning on frozen TRAIN only.

Requires CSE413_Phase1_Baseline.py and CSE413_Phase2_Train.py in the same folder.
Features are the existing 379 biochemical + ESM2-t6 features, with integrity-checked cache.
Selection: mean ROC-AUC on 3 protein-group-disjoint CV folds within TRAIN.
One validation assessment after selection. The sealed TEST partition is never scored.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, ParameterSampler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

import CSE413_Phase1_Baseline as base
import CSE413_Phase2_Train as phase2

OUT = base.ROOT / 'results' / 'phase3_xgboost_tuning'
SEARCH_SPACE = {
    'n_estimators': [250, 350, 500, 700],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.025, 0.04, 0.06, 0.085],
    'min_child_weight': [1, 3, 6],
    'subsample': [0.7, 0.85, 1.0],
    'colsample_bytree': [0.7, 0.85, 1.0],
    'reg_lambda': [1, 3, 8],
    'reg_alpha': [0, 0.1, 0.5],
}

def model_with(params, jobs):
    return XGBClassifier(**params, objective='binary:logistic', eval_metric='logloss',
                         tree_method='hist', n_jobs=jobs, random_state=base.SEED)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trials', type=int, default=15)
    ap.add_argument('--folds', type=int, default=3)
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()
    if args.trials < 1 or args.folds < 2 or args.jobs < 1:
        ap.error('trials/jobs must be positive and folds >= 2')
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if not base.SRC.is_file() or not base.SPLIT.is_file():
        raise FileNotFoundError('Original CSV / frozen split not found')
    csv_hash = base.sha(base.SRC)
    with np.load(base.SPLIT, allow_pickle=False) as z:
        tr, va, te = (z[k].astype(np.int64) for k in ('train','val','test'))
        saved_hash = str(z['csv_sha256'].item())
    if saved_hash != csv_hash:
        raise ValueError('Frozen split hash mismatch: STOP')
    df = pd.read_csv(base.SRC, usecols=['protein_id','protein_sequence','mutant','annotation'], low_memory=False)
    n = len(df)
    if n != 62727 or not df.annotation.isin(['Pathogenic','Benign']).all():
        raise ValueError('Unexpected CSV: STOP')
    if (len(np.unique(np.concatenate([tr,va,te]))) != n
            or sum(map(len, (tr,va,te))) != n
            or min(map(lambda x:x.min(),(tr,va,te))) < 0
            or max(map(lambda x:x.max(),(tr,va,te))) >= n):
        raise ValueError('Invalid split coverage: STOP')
    proteins = df.protein_id.astype(str).to_numpy()
    if set(proteins[tr]) & set(proteins[va]) or set(proteins[tr]) & set(proteins[te]) or set(proteins[va]) & set(proteins[te]):
        raise ValueError('Frozen split contains protein-ID overlap: STOP')
    print(f'FROZEN: train={len(tr)}, validation={len(va)}, SEALED TEST={len(te)}', flush=True)
    print('Verifying ESM cache and generating original biochemical features...', flush=True)
    esm = phase2.load_esm(csv_hash, n)
    bio = base.build_features(df).reset_index(drop=True)
    X = pd.concat([bio, esm], axis=1)
    if len(X) != n or X.columns.duplicated().any() or not np.isfinite(X.to_numpy(dtype=np.float32)).all():
        raise ValueError('Feature matrix invalid: STOP')
    names = X.columns.to_list()
    y = df.annotation.eq('Pathogenic').astype(np.int8).to_numpy()
    # Keep only TRAIN features/labels for CV, validation features/labels for one final assessment.
    Xt = X.iloc[tr].to_numpy(dtype=np.float32, copy=True)
    Xv = X.iloc[va].to_numpy(dtype=np.float32, copy=True)
    yt, yv, gt = y[tr], y[va], proteins[tr]
    del X, esm, bio, df, y
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=base.SEED + 30)
    folds = list(cv.split(Xt, yt, gt))
    for k, (a,b) in enumerate(folds, 1):
        if set(gt[a]) & set(gt[b]) or len(np.unique(yt[b])) != 2 or len(np.unique(yt[a])) != 2:
            raise ValueError(f'CV fold {k} invalid: STOP')
        print(f'CV fold {k}: train={len(a)}, heldout={len(b)}, no shared protein IDs', flush=True)
    params_list = list(ParameterSampler(SEARCH_SPACE, n_iter=args.trials, random_state=base.SEED+31))
    records = []
    trials_path = OUT / 'cv_trial_results.csv'
    for trial, params in enumerate(params_list, 1):
        scores=[]
        t0=time.time()
        for a,b in folds:
            clf=model_with(params,args.jobs)
            clf.fit(Xt[a], yt[a])
            prob=clf.predict_proba(Xt[b])[:,1]
            scores.append(float(roc_auc_score(yt[b],prob)))
        row={'trial':trial,'cv_auc_mean':float(np.mean(scores)),'cv_auc_std':float(np.std(scores)),
             'seconds':round(time.time()-t0,2),**{f'fold_{i+1}_auc':s for i,s in enumerate(scores)}, **params}
        records.append(row)
        pd.DataFrame(records).to_csv(trials_path,index=False)
        print(f'Trial {trial:02d}/{len(params_list)}: grouped-CV AUC={row["cv_auc_mean"]:.5f} '
              f'+/-{row["cv_auc_std"]:.5f} ({row["seconds"]:.1f}s)',flush=True)
    best=max(records,key=lambda x:x['cv_auc_mean'])
    best_params={k:best[k] for k in SEARCH_SPACE}
    print(f'BEST TRAIN-ONLY CV trial={best["trial"]}, mean AUC={best["cv_auc_mean"]:.5f}',flush=True)
    chosen=model_with(best_params,args.jobs)
    chosen.fit(Xt,yt)
    val_prob=chosen.predict_proba(Xv)[:,1]
    met=base.metrics(yv,val_prob)
    print('FINAL SINGLE VALIDATION evaluation:',flush=True)
    print(f'  VAL AUC={met["roc_auc"]:.5f} ACC={met["accuracy"]:.5f} '
          f'F1={met["f1"]:.5f} AP={met["average_precision"]:.5f}',flush=True)
    prior_path=phase2.OUT/'validation_metrics.csv'
    if prior_path.is_file():
        old=pd.read_csv(prior_path)
        old=old.loc[old.model=='XGBoost']
        if len(old)==1:
            x=old.iloc[0]
            print(f'  CHANGE vs Phase2 XGBoost: ACC {(met["accuracy"]-x.accuracy)*100:+.2f} pp; '
                  f'AUC {(met["roc_auc"]-x.roc_auc)*100:+.2f} pp',flush=True)
    joblib.dump(chosen, OUT/'XGBoost_tuned.joblib')
    pd.DataFrame([{'model':'XGBoost_tuned','partition':'validation','n':len(va),
                   'threshold':0.5,**met}]).to_csv(OUT/'validation_metrics.csv',index=False)
    pd.DataFrame({'row_index':va,'protein_id':proteins[va],
                  'y_true':yv,'probability':val_prob}).to_csv(OUT/'validation_predictions.csv',index=False)
    pd.DataFrame({'feature':names,'importance':chosen.feature_importances_}).sort_values(
        'importance',ascending=False).to_csv(OUT/'feature_importance.csv',index=False)
    base.write_json(OUT/'manifest.json', {
        'phase':'CSE413 Phase3 grouped CV tuning','csv_sha256':csv_hash,
        'split_sha256':base.sha(base.SPLIT),'feature_columns':names,
        'model':'XGBoost','seed':base.SEED,'cv':'StratifiedGroupKFold, TRAIN only, grouped by protein_id',
        'trials':len(params_list),'folds':args.folds,'selection_metric':'mean CV ROC-AUC',
        'best_cv_auc_mean':best['cv_auc_mean'],'best_params':best_params,
        'validation_metrics':met,'test_evaluated':False,
        'note':'No homology-level grouping; validation was not used for hyperparameter selection.'})
    print(f'DONE: {OUT} | elapsed {time.time()-start:.1f}s',flush=True)
    print('FINAL TEST REMAINS SEALED.',flush=True)

if __name__=='__main__':
    main()
