import pandas as pd
import numpy as np
import shap
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
AF_FILE = ROOT / 'data' / 'AlphaFold_Features.csv' 
ANN_FILE = ROOT / 'data' / 'UniProt_Annotations.csv'
TOP_K = 300  # আমরা শুধু সেরা ৩০০ ফিচার রাখব

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
    print("Loading All Features (Bio, ESM 650M, AF, Ann)...")
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
    feature_names = np.array(X.columns)
    
    X_train_val = X.iloc[tr_va]
    y_train_val = y[tr_va]
    X_test = X.iloc[te]
    y_test = y[te]  # <--- FIXED: Added y_test here
    del X, esm, bio, af, ann, df

    print("Step 1: Training XGBoost to calculate SHAP importance...")
    xgb_explainer = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, tree_method='hist', random_state=413, n_jobs=8
    )
    xgb_explainer.fit(X_train_val, y_train_val)

    print("Step 2: Extracting SHAP Values and Selecting Top Features...")
    explainer = shap.TreeExplainer(xgb_explainer)
    shap_values = explainer(X_train_val)
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:TOP_K]
    top_features = feature_names[top_indices]
    
    print(f"✅ Selected Top {TOP_K} Features out of {len(feature_names)}")

    X_train_val_filtered = X_train_val[top_features].to_numpy(dtype=np.float32)
    X_test_filtered = X_test[top_features].to_numpy(dtype=np.float32)

    print("Step 3: Retraining Super-Stacking Ensemble with Top Features...")
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
    
    stacking_clf.fit(X_train_val_filtered, y_train_val)

    print("Step 4: Evaluating SHAP-Optimized Model on Sealed Test Set...")
    probs = stacking_clf.predict_proba(X_test_filtered)[:, 1]
    
    opt_thresh = 0.54
    preds = (probs >= opt_thresh).astype(int)
    
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    
    print("\n" + "="*55)
    print(f"✂️ FEATURES REDUCED: 1358 ➡️ {TOP_K}")
    print(f"🚀 SHAP-OPTIMIZED AUROC: {auc:.4f}")
    print(f"🎯 SHAP-OPTIMIZED ACCURACY (Thresh {opt_thresh}): {acc:.4f}")
    print("="*55)

if __name__ == '__main__':
    main()