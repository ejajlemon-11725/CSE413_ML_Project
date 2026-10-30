import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import CSE413_Phase1_Baseline as base

ROOT = Path(r'D:\CSE413_ML_Project')
CACHE = ROOT / 'data' / 'esm2_t33_650M_features'
EXPECTED_ESM_DIMS = 1282

# ==========================================
# AlphaFold & UniProt ফাইলের লোকেশন ঠিক করে দাও
# ==========================================
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
    print("Loading Original Dataset & Frozen Split...")
    df = pd.read_csv(base.SRC, usecols=['protein_id','protein_sequence','mutant','annotation'], low_memory=False)
    n = len(df)
    y = df.annotation.eq('Pathogenic').astype(np.int8).to_numpy()

    with np.load(base.SPLIT, allow_pickle=False) as z:
        tr = z['train'].astype(int)
        va = z['val'].astype(int)

    print("Loading 1. 650M ESM Features...")
    esm = load_esm_650M(n)
    
    print("Loading 2. Biochemical Features...")
    bio = base.build_features(df).reset_index(drop=True)

    print("Loading 3. AlphaFold 3D Features...")
    if AF_FILE.exists():
        af = pd.read_csv(AF_FILE)
    else:
        print("⚠️ AlphaFold CSV not found! Using empty features to maintain pipeline.")
        af = pd.DataFrame(np.nan, index=np.arange(n), columns=[f'af_feat_{i}' for i in range(15)])

    print("Loading 4. UniProt Annotation Features...")
    if ANN_FILE.exists():
        ann = pd.read_csv(ANN_FILE)
    else:
        print("⚠️ UniProt CSV not found! Using empty features to maintain pipeline.")
        ann = pd.DataFrame(np.nan, index=np.arange(n), columns=[f'ann_feat_{i}' for i in range(10)])

    print("Concatenating all feature blocks (Bio + ESM + AF + Ann)...")
    X = pd.concat([bio, esm, af, ann], axis=1)
    
    Xt, yt = X.iloc[tr].to_numpy(dtype=np.float32), y[tr]
    Xv, yv = X.iloc[va].to_numpy(dtype=np.float32), y[va]
    del X, esm, bio, af, ann, df

    print("Setting up Base Models...")
    xgb = XGBClassifier(
        n_estimators=763, max_depth=8, learning_rate=0.06303252725557594,
        subsample=0.7639041423984017, colsample_bytree=0.9404307591524612,
        reg_lambda=6.635419134558022, reg_alpha=0.19104714067076106,
        tree_method='hist', random_state=413, n_jobs=8
    )
    
    lgbm = LGBMClassifier(n_estimators=700, max_depth=8, learning_rate=0.05, random_state=413, n_jobs=8)
    cat = CatBoostClassifier(iterations=700, depth=8, learning_rate=0.05, verbose=0, random_state=413, thread_count=8)

    print("Training Final Super-Stacking Ensemble (This will take a few minutes)...")
    stacking_clf = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)],
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=1 
    )

    stacking_clf.fit(Xt, yt)

    print("Evaluating on Validation Set with ALL Features...")
    probs = stacking_clf.predict_proba(Xv)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    auc = roc_auc_score(yv, probs)
    acc = accuracy_score(yv, preds)
    
    print("\n" + "="*50)
    print(f"🌟 ULTIMATE STACKING AUROC: {auc:.4f}")
    print(f"✅ ULTIMATE STACKING ACCURACY: {acc:.4f}")
    print("="*50)

if __name__ == '__main__':
    main()