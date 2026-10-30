import pandas as pd
import CSE413_Phase1_Baseline as base

def check_dataset_size():
    print("Dataset load kora hocche...")
    
    # Dataset load kora
    df = pd.read_csv(base.SRC, low_memory=False)
    
    total_rows = len(df)
    total_cols = len(df.columns)
    
    print("\n" + "="*50)
    print("📊 DATASET SUMMARY")
    print("="*50)
    print(f"File Path    : {base.SRC}")
    print(f"Total Data (Rows)    : {total_rows}")
    print(f"Total Features (Cols): {total_cols}")
    
    # Pathogenic vs Benign distribution check kora
    if 'annotation' in df.columns:
        print("-" * 50)
        print("Dataset Balance Check:")
        counts = df['annotation'].value_counts()
        for label, count in counts.items():
            percentage = (count / total_rows) * 100
            print(f" - {label:<10}: {count} ({percentage:.2f}%)")
    print("="*50)

if __name__ == '__main__':
    check_dataset_size()