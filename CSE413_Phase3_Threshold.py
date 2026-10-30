import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import CSE413_Phase1_Baseline as base

ROOT = Path(r'D:\CSE413_ML_Project')
CACHE = ROOT / 'data' / 'esm2_t33_650M_features'
EXPECTED_ESM_DIMS = 1282
AF_FILE = ROOT / 'data' / 'AlphaFold_Features.csv' 
ANN_FILE = ROOT / 'data' / 'UniProt_Annotations.csv'

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
    print("Loading Data & Features...")
    df = pd.read_csv(base.SRC, usecols=['protein_id','protein_sequence','mutant','annotation'], low_memory=False)
    n = len(df)
    y = df.annotation.eq('Pathogenic').astype(np.int8).to_numpy()

    with np.load(base.SPLIT, allow_pickle=False) as z:
        tr = z['train'].astype(int)
        va = z['val'].astype(int)
        te = z['test'].astype(int)

    esm = load_esm_650M(n)
    bio = base.build_features(df).reset_index(drop=True)
    af = pd.read_csv(AF_FILE) if AF_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'af_feat_{i}' for i in range(15)])
    ann = pd.read_csv(ANN_FILE) if ANN_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'ann_feat_{i}' for i in range(10)])

    X = pd.concat([bio, esm, af, ann], axis=1)
    
    # Split into Train, Val, Test
    X_train, y_train = X.iloc[tr].to_numpy(dtype=np.float32), y[tr]
    X_val, y_val = X.iloc[va].to_numpy(dtype=np.float32), y[va]
    X_test, y_test = X.iloc[te].to_numpy(dtype=np.float32), y[te]
    del X, esm, bio, af, ann, df

    print("Step 1: Training Stacking Model on Train Set...")
    xgb = XGBClassifier(
        n_estimators=763, max_depth=8, learning_rate=0.06303252725557594,
        subsample=0.7639041423984017, colsample_bytree=0.9404307591524612,
        reg_lambda=6.635419134558022, reg_alpha=0.19104714067076106,
        tree_method='hist', random_state=413, n_jobs=8
    )
    lgbm = LGBMClassifier(n_estimators=700, max_depth=8, learning_rate=0.05, random_state=413, n_jobs=8)
    cat = CatBoostClassifier(iterations=700, depth=8, learning_rate=0.05, verbose=0, random_state=413, thread_count=8)

    stacking_clf = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
        final_estimator=LogisticRegression(),
        cv=3, n_jobs=1 
    )
    
    stacking_clf.fit(X_train, y_train)

    print("Step 2: Finding Optimal Threshold on Validation Set...")
    val_probs = stacking_clf.predict_proba(X_val)[:, 1]
    
    best_thresh = 0.5
    best_acc = 0.0
    
    # Testing thresholds from 0.30 to 0.70
    for thresh in np.arange(0.30, 0.71, 0.01):
        preds = (val_probs >= thresh).astype(int)
        acc = accuracy_score(y_val, preds)
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh
            
    print(f"🎯 Optimal Threshold Found: {best_thresh:.2f} (Val Accuracy: {best_acc:.4f})")

    print("Step 3: Refitting on Train+Val and Evaluating on Sealed Test Set...")
    # Refit on Train+Val for maximum data leverage
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    stacking_clf.fit(X_train_val, y_train_val)
    
    test_probs = stacking_clf.predict_proba(X_test)[:, 1]
    
    # Calculate with old 0.5 vs new optimal threshold
    old_preds = (test_probs >= 0.5).astype(int)
    opt_preds = (test_probs >= best_thresh).astype(int)
    
    print("\n" + "="*55)
    print(f"Default Threshold (0.50) Accuracy: {accuracy_score(y_test, old_preds):.4f}")
    print(f"🚀 OPTIMIZED THRESHOLD ({best_thresh:.2f}) ACCURACY: {accuracy_score(y_test, opt_preds):.4f}")
    print("="*55)

if __name__ == '__main__':
    main()