import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import CSE413_Phase1_Baseline as base

# Setup directories
ROOT = Path(r'D:\CSE413_ML_Project')
CACHE = ROOT / 'data' / 'esm2_t33_650M_features'
AF_FILE = ROOT / 'data' / 'AlphaFold_Features.csv' 
ANN_FILE = ROOT / 'data' / 'UniProt_Annotations.csv'
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(exist_ok=True)

# Set publication style for plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

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

def get_ensemble():
    xgb = XGBClassifier(n_estimators=763, max_depth=8, learning_rate=0.063, random_state=413, n_jobs=8)
    lgbm = LGBMClassifier(n_estimators=700, max_depth=8, learning_rate=0.05, random_state=413, n_jobs=8)
    cat = CatBoostClassifier(iterations=700, depth=8, learning_rate=0.05, verbose=0, random_state=413, thread_count=8)
    return VotingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm), ('cat', cat)], voting='soft', weights=[2, 1, 1])

def main():
    print("Loading Data & Preparing Features...")
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
    model = get_ensemble()
    model.fit(X_train, y[tr_va])
    
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.54).astype(int)

    print("Generating Publication-Ready Figures (300 DPI)...")

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#2c3e50', lw=2.5, label=f'Ensemble AUROC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], color='#e74c3c', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC)', fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'Figure_4A_ROC_Curve.png', dpi=300)
    plt.close()

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color='#2980b9', lw=2.5, label=f'AUPRC = {pr_auc:.4f}')
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'Figure_4B_PR_Curve.png', dpi=300)
    plt.close()

    # 3. Calibration Curve
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)
    plt.figure(figsize=(7, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", color='#8e44ad', lw=2.5, label="Hybrid Ensemble")
    plt.plot([0, 1], [0, 1], "k:", lw=2, label="Perfect Calibration")
    plt.xlabel('Mean Predicted Probability', fontweight='bold')
    plt.ylabel('Fraction of Positives', fontweight='bold')
    plt.title('Reliability Diagram / Calibration Curve', fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'Figure_5_Calibration.png', dpi=300)
    plt.close()

    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Benign', 'Pathogenic'], yticklabels=['Benign', 'Pathogenic'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Confusion Matrix (Threshold = 0.54)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'Figure_S2_Confusion_Matrix.png', dpi=300)
    plt.close()

    # 5. Risk-Coverage Curve
    thresholds = np.linspace(0.5, 0.99, 50)
    risks, coverages = [], []
    conf_scores = np.maximum(probs, 1 - probs)
    correct_preds = (preds == y_test)
    for t in thresholds:
        mask = conf_scores >= t
        coverage = np.mean(mask)
        risk = 1 - np.mean(correct_preds[mask]) if coverage > 0 else 0
        risks.append(risk)
        coverages.append(coverage)
        
    plt.figure(figsize=(7, 6))
    plt.plot(np.array(coverages)*100, np.array(risks)*100, color='#d35400', lw=2.5)
    plt.xlabel('Coverage (%)', fontweight='bold')
    plt.ylabel('Risk (Error Rate %)', fontweight='bold')
    plt.title('Risk-Coverage Trade-off', fontweight='bold')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'Figure_7_Risk_Coverage.png', dpi=300)
    plt.close()

    # 6. Feature Importance (Using LightGBM proxy for speed)
    lgbm_model = model.named_estimators_['lgbm']
    importances = lgbm_model.feature_importances_
    indices = np.argsort(importances)[-20:] # Top 20
    top_feat_names = [top_features[i] for i in indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), importances[indices], color='#27ae60', align='center')
    plt.yticks(range(20), top_feat_names)
    plt.xlabel('LightGBM Feature Importance (Split)', fontweight='bold')
    plt.title('Top 20 Features Driving Predictions', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'Figure_6_Feature_Importance.png', dpi=300)
    plt.close()

    print(f"✅ SUCCESS! All publication-ready figures saved in: {FIG_DIR}")

if __name__ == '__main__':
    main()