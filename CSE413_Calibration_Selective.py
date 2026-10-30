import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve
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
    print("Loading Data for Calibration Analysis...")
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
    
    # Model Training
    print("Training Model...")
    model = LGBMClassifier(n_estimators=700, max_depth=8, learning_rate=0.05, random_state=413, n_jobs=8)
    model.fit(X_train, y[tr_va])
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.54).astype(int)
    
    print("\n" + "="*65)
    print("📊 1. CONFIDENCE CALIBRATION METRICS")
    print("="*65)
    
    # Brier Score Calculation (Lower is better, 0 is perfect)
    brier = brier_score_loss(y_test, probs)
    print(f"Brier Score: {brier:.4f} (Closer to 0 means highly calibrated)")
    
    # Reliability Diagram (Calibration Curve)
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model Calibration")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.ylabel("Fraction of Positives (Actual True %)")
    plt.xlabel("Mean Predicted Value (Model's Confidence %)")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend()
    plt.grid(True)
    plt.savefig(ROOT / 'Reliability_Diagram.png')
    print("✅ Reliability Diagram saved as 'Reliability_Diagram.png'")
    
    print("\n" + "="*65)
    print("🛡️ 2. SELECTIVE PREDICTION (ABSTENTION ANALYSIS)")
    print("="*65)
    
    # Confidence calculation: Distance from 0.5
    confidence = np.maximum(probs, 1 - probs)
    
    abstention_thresholds = [0.60, 0.70, 0.80, 0.90]
    
    print(f"{'Conf. Threshold':<18} | {'Predicted (Confident)':<22} | {'Abstained (Review)':<20} | {'Confident Acc.'}")
    print("-" * 85)
    
    for th in abstention_thresholds:
        confident_mask = confidence >= th
        unsure_mask = ~confident_mask
        
        predicted_pct = np.mean(confident_mask) * 100
        abstained_pct = np.mean(unsure_mask) * 100
        
        if np.sum(confident_mask) > 0:
            conf_acc = accuracy_score(y_test[confident_mask], preds[confident_mask])
            print(f"Conf >= {th:.2f} {'':<5} | {predicted_pct:>6.2f}% Data {'':<7} | {abstained_pct:>6.2f}% Flagged {'':<5} | {conf_acc:.4f}")
        else:
            print(f"Conf >= {th:.2f} {'':<5} | {predicted_pct:>6.2f}% Data {'':<7} | {abstained_pct:>6.2f}% Flagged {'':<5} | N/A")
            
    print("="*65)

if __name__ == '__main__':
    main()