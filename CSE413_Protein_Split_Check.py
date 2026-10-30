import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import CSE413_Phase1_Baseline as base
from pathlib import Path

def verify_protein_wise_split():
    print("Dataset load kora hocche...")
    # Dataset theke shudhu dorkari column gulo neya hocche
    df = pd.read_csv(base.SRC, usecols=['protein_id', 'mutant', 'annotation'], low_memory=False)
    
    print("\nProtein-wise (Homology) split kora hocche...")
    # protein_id ke group dhore 85% Train ebong 15% Test e bhag kora hocche
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=413)
    
    train_idx, test_idx = next(gss.split(df, df['annotation'], groups=df['protein_id']))
    
    # Train ar Test set er protein ID gulo alada kora
    train_proteins = set(df.iloc[train_idx]['protein_id'])
    test_proteins = set(df.iloc[test_idx]['protein_id'])
    
    # Kono protein train ar test e eksathe poreche kina tar overlap check
    overlap = train_proteins.intersection(test_proteins)
    
    print("-" * 50)
    print(f"Total Rows: Train={len(train_idx)}, Test={len(test_idx)}")
    print(f"Unique Proteins in Train Set: {len(train_proteins)}")
    print(f"Unique Proteins in Test Set:  {len(test_proteins)}")
    print(f"Overlapping Proteins (Leakage): {len(overlap)}")
    print("-" * 50)
    
    if len(overlap) == 0:
        print("✅ SUCCESS: Strict Protein-wise split verified! Kono data leakage nei.")
    else:
        print(f"❌ ERROR: Data leakage pawa geche! Overlapping proteins: {len(overlap)}")

if __name__ == '__main__':
    verify_protein_wise_split()