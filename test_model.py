import pandas as pd
import joblib
import numpy as np

# ডেটাসেটের সঠিক পাথ
file_name = "data/Clinical_substitutions.csv" 

try:
    print(f"Loading data from {file_name}...")
    df = pd.read_csv(file_name)
    
    print("Loading models...")
    model = joblib.load("final_stacking_model.pkl")
    top_features = joblib.load("top_300_features.pkl")
    
    # Feature filtering
    for col in top_features:
        if col not in df.columns:
            df[col] = 0
            
    X_input = df[top_features].to_numpy(dtype=np.float32)
    
    print("Calculating probabilities...")
    probs = model.predict_proba(X_input)[:, 1]
    
    print("\n--- Model Probability Analysis ---")
    print(f"Max Probability: {np.max(probs):.4f}")
    print(f"Min Probability: {np.min(probs):.4f}")
    print(f"Mean Probability: {np.mean(probs):.4f}")
    print(f"Total Pathogenic (>= 0.54): {np.sum(probs >= 0.54)}")
    print(f"Total Benign (< 0.54): {np.sum(probs < 0.54)}")
    print("----------------------------------\n")
    
except Exception as e:
    print(f"Error: {e}")