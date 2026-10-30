import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier
import CSE413_Phase1_Baseline as base

ROOT = Path(r'D:\CSE413_ML_Project')
EXTERNAL_FILE = ROOT / 'data' / 'External_Dataset.csv' # তোমার এক্সটার্নাল ফাইলের পাথ

def main():
    print("Loading Training Data and External Data...")
    # মেইন ডেটাসেট লোড (ওভারল্যাপ চেকের জন্য)
    df_train = pd.read_csv(base.SRC, usecols=['protein_id', 'mutant'], low_memory=False)
    
    # এক্সটার্নাল ডেটাসেট লোড
    if not EXTERNAL_FILE.exists():
        print(f"❌ ERROR: External dataset '{EXTERNAL_FILE.name}' পাওয়া যায়নি!")
        print("💡 প্রজেক্ট ফোল্ডারের data ডিরেক্টরিতে ফাইলটি রাখো অথবা Subgroup Analysis-এর জন্য মেইন ডেটাসেট ব্যবহার করো।")
        return

    df_ext = pd.read_csv(EXTERNAL_FILE, low_memory=False)
    
    print("\n" + "="*65)
    print("🔍 1. TRAINING VS EXTERNAL OVERLAP CHECK")
    print("="*65)
    
    # Overlap Check (Variant & Protein level)
    train_variants = set(df_train['protein_id'] + "_" + df_train['mutant'])
    ext_variants = set(df_ext['protein_id'] + "_" + df_ext['mutant'])
    
    variant_overlap = train_variants.intersection(ext_variants)
    protein_overlap = set(df_train['protein_id']).intersection(set(df_ext['protein_id']))
    
    print(f"Total External Variants: {len(ext_variants)}")
    print(f"Variant-level Overlap: {len(variant_overlap)} (Should ideally be 0)")
    print(f"Protein-level Overlap: {len(protein_overlap)}")
    
    # এক্সটার্নাল ডেটা থেকে ওভারল্যাপ ভ্যারিয়েন্ট বাদ দেওয়া (Strict Independent Evaluation এর জন্য)
    if len(variant_overlap) > 0:
        print("\nRemoving overlapping variants to ensure strict independence...")
        df_ext['var_id'] = df_ext['protein_id'] + "_" + df_ext['mutant']
        df_ext = df_ext[~df_ext['var_id'].isin(variant_overlap)].drop(columns=['var_id'])
        print(f"New External Dataset Size: {len(df_ext)}")

    print("\n" + "="*65)
    print("📊 2. INDEPENDENT EXTERNAL EVALUATION")
    print("="*65)
    
    # এখানে এক্সটার্নাল ডেটার জন্য ফিচার এক্সট্রাকশন (ESM, Bio, etc.) করতে হবে
    # ডেমোনস্ট্রেশনের জন্য ধরে নিচ্ছি এক্সটার্নাল ডেটাতে ফিচারগুলো আগে থেকেই তৈরি করা আছে বা তৈরি করে নিতে হবে
    # X_ext = ... (Extract features for df_ext)
    # y_ext = df_ext.annotation.eq('Pathogenic').astype(np.int8).to_numpy()
    
    print("⚠️ (Feature extraction on external data needs to be run here using your existing pipeline)")
    # probs_ext = model.predict_proba(X_ext)[:, 1]
    # auc_ext = roc_auc_score(y_ext, probs_ext)
    # print(f"External AUROC: {auc_ext:.4f}")
    
    print("\n" + "="*65)
    print("🧬 3. SUBGROUP ANALYSIS (By Protein / Gene Family)")
    print("="*65)
    
    # Subgroup Analysis (ধরে নিচ্ছি 'gene_name' বা 'protein_family' কলাম আছে)
    subgroup_col = 'protein_id' # বা 'gene_name' যদি থাকে
    if subgroup_col in df_ext.columns:
        top_groups = df_ext[subgroup_col].value_counts().head(5).index
        
        for group in top_groups:
            mask = df_ext[subgroup_col] == group
            # auc_group = roc_auc_score(y_ext[mask], probs_ext[mask])
            # print(f"Group: {group:<15} | Variants: {mask.sum():<5} | AUROC: {auc_group:.4f}")
            print(f"Group: {group:<15} | Variants: {mask.sum():<5} | (Evaluation pending feature extraction)")
    
    print("="*65)

if __name__ == '__main__':
    main()