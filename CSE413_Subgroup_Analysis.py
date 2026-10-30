import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
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
    print("Loading Data for Subgroup Analysis...")
    df = pd.read_csv(base.SRC, usecols=['protein_id', 'protein_sequence', 'mutant', 'annotation'], low_memory=False)
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
    
    X_train = X.iloc[tr_va][top_features].to_numpy(dtype=np.float32)
    X_test = X.iloc[te][top_features].to_numpy(dtype=np.float32)
    y_test = y[te]
    df_test = df.iloc[te].copy()

    print("Training Model...")
    model = LGBMClassifier(n_estimators=700, max_depth=8, learning_rate=0.05, random_state=413, n_jobs=8)
    model.fit(X_train, y[tr_va])
    
    df_test['probs'] = model.predict_proba(X_test)[:, 1]
    df_test['preds'] = (df_test['probs'] >= 0.54).astype(int)
    df_test['y_true'] = y_test

    print("\n" + "="*70)
    print("🧬 SUBGROUP ANALYSIS (Top 10 Proteins with most variants in Test Set)")
    print("="*70)
    
    # টেস্ট সেটের টপ ১০টি প্রোটিন বের করা
    top_proteins = df_test['protein_id'].value_counts().head(10).index
    
    print(f"{'Protein ID':<15} | {'Total Variants':<15} | {'Accuracy':<10} | {'AUROC'}")
    print("-" * 70)
    
    for pid in top_proteins:
        mask = df_test['protein_id'] == pid
        sub_df = df_test[mask]
        
        acc = accuracy_score(sub_df['y_true'], sub_df['preds'])
        
        # AUROC ক্যালকুলেট করার জন্য প্রোটিনটিতে কমপক্ষে একটি পজিটিভ এবং একটি নেগেটিভ ক্লাস থাকতে হবে
        if len(np.unique(sub_df['y_true'])) > 1:
            auc = roc_auc_score(sub_df['y_true'], sub_df['probs'])
            auc_str = f"{auc:.4f}"
        else:
            auc_str = "N/A (One Class)"
            
        print(f"{pid:<15} | {len(sub_df):<15} | {acc:.4f}     | {auc_str}")

    print("="*70)

if __name__ == '__main__':
    main()