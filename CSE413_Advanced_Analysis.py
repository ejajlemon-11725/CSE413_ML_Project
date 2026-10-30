import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import CSE413_Phase1_Baseline as base

ROOT = Path(r'D:\CSE413_ML_Project')
CACHE = ROOT / 'data' / 'esm2_t33_650M_features'
AF_FILE = ROOT / 'data' / 'AlphaFold_Features.csv' 
ANN_FILE = ROOT / 'data' / 'UniProt_Annotations.csv'

def load_esm_650M(n):
    shards = sorted(CACHE.glob('rows_*.npz'))
    X = np.empty((n, 1282), dtype=np.float32)
    next_start = 0
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            chunk = z['X']
            end = next_start + len(chunk)
            X[next_start:end] = chunk
            next_start = end
    cols = [f'esm_delta_{j:03d}' for j in range(1280)] + ['esm_delta_l2', 'esm_wt_mut_cosine']
    return pd.DataFrame(X, columns=cols)

def get_base_models():
    xgb = XGBClassifier(
        n_estimators=763, max_depth=8, learning_rate=0.063,
        subsample=0.76, colsample_bytree=0.94,
        reg_lambda=6.63, reg_alpha=0.19,
        tree_method='hist', random_state=413, n_jobs=8
    )
    lgbm = LGBMClassifier(n_estimators=700, max_depth=8, learning_rate=0.05, random_state=413, n_jobs=8)
    cat = CatBoostClassifier(iterations=700, depth=8, learning_rate=0.05, verbose=0, random_state=413, thread_count=8)
    return VotingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)], voting='soft', weights=[2, 1, 1])

def main():
    print("Loading Data and Splitting...")
    df = pd.read_csv(base.SRC, usecols=['protein_id','protein_sequence','mutant','annotation'], low_memory=False)
    n = len(df)
    y = df.annotation.eq('Pathogenic').astype(np.int8).to_numpy()

    with np.load(base.SPLIT, allow_pickle=False) as z:
        tr_va = np.concatenate([z['train'].astype(int), z['val'].astype(int)])
        te = z['test'].astype(int)

    esm = load_esm_650M(n)
    bio = base.build_features(df).reset_index(drop=True)
    af = pd.read_csv(AF_FILE) if AF_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'af_feat_{i}' for i in range(15)])
    ann = pd.read_csv(ANN_FILE) if ANN_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'ann_feat_{i}' for i in range(10)])

    X = pd.concat([bio, esm, af, ann], axis=1)
    top_features = joblib.load(ROOT / 'top_300_features.pkl')
    
    # ফিচারগুলোকে তোমার নির্দেশ অনুযায়ী গ্রুপ করা হচ্ছে
    bio_cols = [f for f in top_features if not any(x in f for x in ['esm', 'af_', 'ann_'])]
    esm_cols = [f for f in top_features if 'esm' in f]
    af_cols = [f for f in top_features if 'af_' in f]
    
    # ৫টি ধাপের Ablation Ladder তৈরি
    steps = [
        ("Step 1: Biochemical/Context Only", bio_cols),
        ("Step 2: ESM-2 Features Only", esm_cols),
        ("Step 3: ESM-2 + Biochemical/Context", bio_cols + esm_cols),
        ("Step 4: ESM-2 + AlphaFold + Bio", bio_cols + esm_cols + af_cols),
        ("Step 5: All Features + Ensemble", top_features)
    ]
    
    print("\n" + "="*65)
    print("🚀 5-STEP ABLATION STUDY ANALYSIS")
    print("="*65)
    
    for step_name, cols in steps:
        if len(cols) == 0:
            print(f"{step_name:<35} | SKIPPED (No features found)")
            continue
            
        X_train = X.iloc[tr_va][cols].to_numpy(dtype=np.float32)
        X_test = X.iloc[te][cols].to_numpy(dtype=np.float32)
        
        # আমরা শুধুমাত্র LightGBM ব্যবহার করছি প্রথম ৪টি ধাপ দ্রুত করার জন্য,
        # আর শেষ ধাপে ফুল Voting Ensemble ব্যবহার করা হচ্ছে (Step 5)
        if "Ensemble" in step_name:
            model = get_base_models()
        else:
            model = LGBMClassifier(n_estimators=700, max_depth=8, learning_rate=0.05, random_state=413, n_jobs=8)
            
        model.fit(X_train, y[tr_va])
        probs = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y[te], probs)
        acc = accuracy_score(y[te], (probs >= 0.54).astype(int))
        print(f"{step_name:<35} | AUROC: {auc:.4f} | Accuracy: {acc:.4f}")
        
    print("="*65)

if __name__ == '__main__':
    main()