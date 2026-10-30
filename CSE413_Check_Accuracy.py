import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
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

def main():
    print("Loading Data and extracting test set...")
    df = pd.read_csv(base.SRC, usecols=['protein_id','protein_sequence','mutant','annotation'], low_memory=False)
    n = len(df)
    y = df.annotation.eq('Pathogenic').astype(np.int8).to_numpy()

    with np.load(base.SPLIT, allow_pickle=False) as z:
        te = z['test'].astype(int)

    esm = load_esm_650M(n)
    bio = base.build_features(df).reset_index(drop=True)
    af = pd.read_csv(AF_FILE) if AF_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'af_feat_{i}' for i in range(15)])
    ann = pd.read_csv(ANN_FILE) if ANN_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'ann_feat_{i}' for i in range(10)])

    X = pd.concat([bio, esm, af, ann], axis=1)
    
    print("Loading Saved Model & Top Features...")
    top_features = joblib.load(ROOT / 'top_300_features.pkl')
    model = joblib.load(ROOT / 'final_stacking_model.pkl')

    X_test_filtered = X.iloc[te][top_features].to_numpy(dtype=np.float32)
    y_test = y[te]

    print("Evaluating...")
    probs = model.predict_proba(X_test_filtered)[:, 1]
    
    # 0.54 threshold as optimized earlier
    opt_thresh = 0.54
    preds = (probs >= opt_thresh).astype(int)
    
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    
    print("\n" + "="*55)
    print(f"✅ DEPLOYED MODEL AUROC: {auc:.4f}")
    print(f"🎯 DEPLOYED MODEL ACCURACY (Thresh {opt_thresh}): {acc:.4f}")
    print("="*55)

if __name__ == '__main__':
    main()