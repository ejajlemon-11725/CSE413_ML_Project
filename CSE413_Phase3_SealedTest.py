import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
import joblib
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
    print("Loading Original Dataset & Frozen Split...")
    df = pd.read_csv(base.SRC, usecols=['protein_id','protein_sequence','mutant','annotation'], low_memory=False)
    n = len(df)
    y = df.annotation.eq('Pathogenic').astype(np.int8).to_numpy()

    with np.load(base.SPLIT, allow_pickle=False) as z:
        # Combining train and val for the final refit
        tr_va = np.concatenate([z['train'].astype(int), z['val'].astype(int)])
        te = z['test'].astype(int)

    print("Loading All Feature Blocks (Bio, ESM 650M, AF, Ann)...")
    esm = load_esm_650M(n)
    bio = base.build_features(df).reset_index(drop=True)
    af = pd.read_csv(AF_FILE) if AF_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'af_feat_{i}' for i in range(15)])
    ann = pd.read_csv(ANN_FILE) if ANN_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'ann_feat_{i}' for i in range(10)])

    X = pd.concat([bio, esm, af, ann], axis=1)
    
    # ----------------------------------------------------
    # ADDED FOR STREAMLIT: Save the actual test data
    # ----------------------------------------------------
    print("Saving realistic biological test data for Streamlit...")
    try:
        top_features = joblib.load('top_300_features.pkl')
        df_test_real = X.iloc[te].copy()
        
        df_test_300 = pd.DataFrame(index=df_test_real.index)
        for c in top_features:
            if c in df_test_real.columns:
                df_test_300[c] = df_test_real[c]
            else:
                df_test_300[c] = 0 # Safety fallback
                
        # Get first 200 variants (a good mix of Benign/Pathogenic) for the dashboard
        df_test_300.head(200).to_csv("final_real_test_data.csv", index=False)
        print("✅ Success! 'final_real_test_data.csv' is ready for Streamlit!")
    except Exception as e:
        print(f"⚠️ Could not save CSV. Error: {e}")
    # ----------------------------------------------------

    X_train_val, y_train_val = X.iloc[tr_va].to_numpy(dtype=np.float32), y[tr_va]
    X_test, y_test = X.iloc[te].to_numpy(dtype=np.float32), y[te]
    
    # Delete massive variables to free up RAM
    del X, esm, bio, af, ann, df

    print("Training Final Super-Stacking Ensemble on Train+Val (Might take a bit longer)...")
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
        cv=3,
        n_jobs=1 
    )

    stacking_clf.fit(X_train_val, y_train_val)

    print("Opening the Sealed Test Set for Final Evaluation...")
    probs = stacking_clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    
    print("\n" + "="*55)
    print(f"🏆 SEALED TEST AUROC: {auc:.4f}")
    print(f"🎯 SEALED TEST ACCURACY: {acc:.4f}")
    print(f"📊 SEALED TEST MCC: {mcc:.4f}")
    print("="*55)

if __name__ == '__main__':
    main()