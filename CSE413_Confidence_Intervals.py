import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
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
    print("Loading Data for Bootstrapping Analysis...")
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

    print("Training Final Ensemble Model...")
    model = get_base_models()
    model.fit(X_train, y[tr_va])
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.54).astype(int)

    print("\n" + "="*65)
    print("📈 95% CONFIDENCE INTERVALS (1000 BOOTSTRAP ITERATIONS)")
    print("="*65)
    
    n_bootstraps = 1000
    rng = np.random.RandomState(413)
    
    metrics = {'AUROC': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_test), len(y_test))
        if len(np.unique(y_test[indices])) < 2:
            continue
            
        metrics['AUROC'].append(roc_auc_score(y_test[indices], probs[indices]))
        metrics['Accuracy'].append(accuracy_score(y_test[indices], preds[indices]))
        metrics['Precision'].append(precision_score(y_test[indices], preds[indices]))
        metrics['Recall'].append(recall_score(y_test[indices], preds[indices]))
        metrics['F1-Score'].append(f1_score(y_test[indices], preds[indices]))

    print(f"{'Metric':<15} | {'Mean Value':<12} | {'95% Confidence Interval'}")
    print("-" * 65)
    
    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        print(f"{metric_name:<15} | {mean_val:.4f}       | [{lower:.4f} - {upper:.4f}]")
        
    print("="*65)

if __name__ == '__main__':
    main()