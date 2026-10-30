import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path
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
TOP_K = 300

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
    print("Loading All Data for Final Model Generation...")
    df = pd.read_csv(base.SRC, usecols=['protein_id','protein_sequence','mutant','annotation'], low_memory=False)
    n = len(df)
    y = df.annotation.eq('Pathogenic').astype(np.int8).to_numpy()

    esm = load_esm_650M(n)
    bio = base.build_features(df).reset_index(drop=True)
    af = pd.read_csv(AF_FILE) if AF_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'af_feat_{i}' for i in range(15)])
    ann = pd.read_csv(ANN_FILE) if ANN_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'ann_feat_{i}' for i in range(10)])

    X = pd.concat([bio, esm, af, ann], axis=1)
    feature_names = np.array(X.columns)
    
    # ডেপ্লয়মেন্টের জন্য সম্পূর্ণ ডেটাসেট (Train+Val+Test) ব্যবহার করা বেস্ট প্র্যাকটিস
    X_full = X.copy()
    y_full = y.copy()
    del X, esm, bio, af, ann, df

    print("Step 1: Calculating SHAP Importance to extract Top 300 Features...")
    xgb_explainer = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, tree_method='hist', random_state=413, n_jobs=8
    )
    xgb_explainer.fit(X_full, y_full)

    explainer = shap.TreeExplainer(xgb_explainer)
    shap_values = explainer(X_full)
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:TOP_K]
    top_features = feature_names[top_indices]
    
    print("Saving Feature List...")
    joblib.dump(top_features, ROOT / 'top_300_features.pkl')

    X_full_filtered = X_full[top_features].to_numpy(dtype=np.float32)

    print("Step 2: Training Final Production Model with Top 300 Features...")
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
    
    stacking_clf.fit(X_full_filtered, y_full)

    print("Saving Production Model...")
    joblib.dump(stacking_clf, ROOT / 'final_stacking_model.pkl')
    
    print("\n✅ SUCCESS! 'final_stacking_model.pkl' and 'top_300_features.pkl' are ready for deployment.")

if __name__ == '__main__':
    main()