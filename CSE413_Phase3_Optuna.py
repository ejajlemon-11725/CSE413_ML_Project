import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import CSE413_Phase1_Baseline as base

ROOT = Path(r'D:\CSE413_ML_Project')
# Pointing to the new 650M features directory
CACHE = ROOT / 'data' / 'esm2_t33_650M_features'
EXPECTED_ESM_DIMS = 1282

def load_esm_650M(n):
    shards = sorted(CACHE.glob('rows_*.npz'))
    X = np.empty((n, EXPECTED_ESM_DIMS), dtype=np.float32)
    next_start = 0
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            chunk = z['X']
            end = next_start + len(chunk)
            X[next_start:end] = chunk
            next_start = end
    cols = [f'esm_delta_{j:03d}' for j in range(1280)] + ['esm_delta_l2', 'esm_wt_mut_cosine']
    return pd.DataFrame(X, columns=cols)

def main():
    print("Loading Original Dataset & Frozen Split...")
    df = pd.read_csv(base.SRC, usecols=['protein_id','protein_sequence','mutant','annotation'], low_memory=False)
    n = len(df)
    y = df.annotation.eq('Pathogenic').astype(np.int8).to_numpy()
    groups = df.protein_id.astype(str).to_numpy()

    with np.load(base.SPLIT, allow_pickle=False) as z:
        tr = z['train'].astype(int)

    print("Loading 650M ESM Features (This might take a moment)...")
    esm = load_esm_650M(n)
    print("Building Biochemical Features...")
    bio = base.build_features(df).reset_index(drop=True)

    # Combine 51 Bio + 1282 ESM features = 1333 Features
    X = pd.concat([bio, esm], axis=1)
    
    # We only use the frozen TRAIN partition for CV tuning
    Xt = X.iloc[tr].to_numpy(dtype=np.float32)
    yt = y[tr]
    gt = groups[tr]

    del X, esm, bio, df

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'tree_method': 'hist',
            'random_state': 413,
            'n_jobs': 8
        }

        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=413)
        scores = []
        for a, b in cv.split(Xt, yt, gt):
            clf = XGBClassifier(**params)
            clf.fit(Xt[a], yt[a])
            preds = clf.predict_proba(Xt[b])[:, 1]
            scores.append(roc_auc_score(yt[b], preds))
        
        auc = np.mean(scores)
        return auc

    print("Starting Optuna Optimization (30 Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*50)
    print(f"🏆 BEST TRIAL AUROC: {study.best_value:.4f}")
    print("="*50)
    print(f"Best Parameters to use in final model:\n{study.best_params}")

if __name__ == '__main__':
    main()