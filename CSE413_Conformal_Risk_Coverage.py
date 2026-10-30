import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
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
    print("Loading Data for Conformal Prediction...")
    df = pd.read_csv(base.SRC, usecols=['protein_id', 'protein_sequence', 'mutant', 'annotation'], low_memory=False)
    n = len(df)
    y = df.annotation.eq('Pathogenic').astype(np.int8).to_numpy()

    # Train, Validation (Calibration), এবং Test সেট আলাদা করা
    with np.load(base.SPLIT, allow_pickle=False) as z:
        tr = z['train'].astype(int)
        va = z['val'].astype(int)
        te = z['test'].astype(int)

    esm = load_esm_650M(n)
    bio = base.build_features(df).reset_index(drop=True)
    af = pd.read_csv(AF_FILE) if AF_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'af_feat_{i}' for i in range(15)])
    ann = pd.read_csv(ANN_FILE) if ANN_FILE.exists() else pd.DataFrame(np.nan, index=np.arange(n), columns=[f'ann_feat_{i}' for i in range(10)])

    X = pd.concat([bio, esm, af, ann], axis=1)
    top_features = joblib.load(ROOT / 'top_300_features.pkl')
    
    # ডেটা স্লাইসিং
    X_train = X.iloc[tr][top_features].to_numpy(dtype=np.float32)
    y_train = y[tr]
    
    X_calib = X.iloc[va][top_features].to_numpy(dtype=np.float32)
    y_calib = y[va]
    
    X_test = X.iloc[te][top_features].to_numpy(dtype=np.float32)
    y_test = y[te]

    print("Training Model on Train Split...")
    model = LGBMClassifier(n_estimators=700, max_depth=8, learning_rate=0.05, random_state=413, n_jobs=8)
    model.fit(X_train, y_train)

    print("\n" + "="*75)
    print("🛡️ 1. CONFORMAL PREDICTION (Marginal Coverage Guarantee)")
    print("="*75)
    
    # Calibration set দিয়ে Non-conformity score বের করা
    calib_probs = model.predict_proba(X_calib)
    calib_scores = 1 - calib_probs[np.arange(len(y_calib)), y_calib]
    
    alpha_levels = [0.05, 0.10, 0.15] # Target Error Rate (5%, 10%, 15%)
    test_probs = model.predict_proba(X_test)
    
    print(f"{'Target Error (Alpha)':<22} | {'Actual Error':<15} | {'Average Set Size'}")
    print("-" * 75)
    
    for alpha in alpha_levels:
        n_calib = len(calib_scores)
        q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        q_level = min(q_level, 1.0)
        
        q_hat = np.quantile(calib_scores, q_level)
        
        # Test সেটের জন্য Prediction Set তৈরি করা
        prediction_sets = test_probs >= (1 - q_hat)
        
        # কভারেজ চেক: আসল ক্লাসটি Prediction Set এর ভেতর আছে কি না?
        covered = prediction_sets[np.arange(len(y_test)), y_test]
        actual_error = 1 - np.mean(covered)
        avg_set_size = np.mean(np.sum(prediction_sets, axis=1))
        
        print(f"Target: {alpha*100:5.1f}% {'':<7} | {actual_error*100:7.2f}% {'':<7} | {avg_set_size:.3f} classes")

    print("\n" + "="*75)
    print("📉 2. RISK-COVERAGE ANALYSIS")
    print("="*75)
    
    confidence_scores = np.max(test_probs, axis=1)
    preds = np.argmax(test_probs, axis=1)
    correct_preds = (preds == y_test)
    
    thresholds = np.linspace(0.5, 0.99, 50)
    risks, coverages = [], []
    
    print(f"{'Conf. Threshold':<18} | {'Coverage (%)':<15} | {'Risk (Error %)'}")
    print("-" * 75)
    
    display_thresholds = [0.60, 0.70, 0.80, 0.90, 0.95]
    
    for t in thresholds:
        mask = confidence_scores >= t
        coverage = np.mean(mask)
        risk = 1 - np.mean(correct_preds[mask]) if coverage > 0 else 0
            
        risks.append(risk)
        coverages.append(coverage)
        
        # নির্দিষ্ট থ্রেশহোল্ডগুলো টার্মিনালে প্রিন্ট করার জন্য
        if any(np.isclose(t, dt, atol=0.01) for dt in display_thresholds):
            print(f">= {t:.2f} {'':<11} | {coverage*100:7.2f}% {'':<7} | {risk*100:7.2f}%")

    # Risk-Coverage গ্রাফ তৈরি ও সেভ করা
    plt.figure(figsize=(8, 6))
    plt.plot(np.array(coverages)*100, np.array(risks)*100, 'b-', linewidth=2.5)
    plt.xlabel('Coverage (% of Test Data Predicted)', fontsize=12)
    plt.ylabel('Risk (Error Rate %)', fontsize=12)
    plt.title('Risk-Coverage Trade-off Curve', fontsize=14, pad=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(ROOT / 'Risk_Coverage_Curve.png', dpi=300)
    print("\n✅ Risk-Coverage Curve saved as 'Risk_Coverage_Curve.png'")
    print("="*75)

if __name__ == '__main__':
    main()