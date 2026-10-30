import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
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
        # We'll train on train+val and explain the test set predictions
        tr_va = np.concatenate([z['train'].astype(int), z['val'].astype(int)])
        te = z['test'].astype(int)

    esm = load_esm_650M(n)
    bio = base.build_features(df).reset_index(drop=True)
    af = pd.read_csv(AF_FILE) if AF_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'af_feat_{i}' for i in range(15)])
    ann = pd.read_csv(ANN_FILE) if ANN_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'ann_feat_{i}' for i in range(10)])

    X = pd.concat([bio, esm, af, ann], axis=1)
    feature_names = X.columns.tolist()
    
    X_train_val, y_train_val = X.iloc[tr_va], y[tr_va]
    X_test = X.iloc[te]

    print("Training Best XGBoost Model for SHAP Explanation...")
    xgb = XGBClassifier(
        n_estimators=763, max_depth=8, learning_rate=0.06303252725557594,
        subsample=0.7639041423984017, colsample_bytree=0.9404307591524612,
        reg_lambda=6.635419134558022, reg_alpha=0.19104714067076106,
        tree_method='hist', random_state=413, n_jobs=8
    )
    xgb.fit(X_train_val, y_train_val)

    print("Calculating SHAP values (TreeSHAP is very fast)...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X_test)

    print("Generating SHAP Summary Plot...")
    # Plot top 20 most important features
    shap.summary_plot(shap_values, X_test, max_display=20, show=False)
    
    # Save the plot
    out_path = ROOT / "shap_summary_plot.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ SHAP plot saved successfully at: {out_path}")

if __name__ == '__main__':
    main()