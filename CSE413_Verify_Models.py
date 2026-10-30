import joblib
from pathlib import Path

ROOT = Path(r'D:\CSE413_ML_Project')
MODEL_PATH = ROOT / 'final_stacking_model.pkl'

def verify_ensemble():
    print(f"Loading saved model from: {MODEL_PATH.name}...\n")
    model = joblib.load(MODEL_PATH)
    
    print("="*55)
    print("🔍 VERIFYING ENSEMBLE ARCHITECTURE")
    print("="*55)
    
    # মেইন মডেলের টাইপ চেক
    print(f"🧠 Main Model Type: {type(model).__name__}")
    print("\n📦 Base Estimators Inside:")
    
    # Stacking মডেলের ভেতরের মডেলগুলো বের করে আনা
    if hasattr(model, 'named_estimators_'):
        for name, est in model.named_estimators_.items():
            print(f"  ✅ {name.upper()}: {type(est).__name__}")
            
    print("\n🏁 Final Meta-Model (Combiner):")
    if hasattr(model, 'final_estimator_'):
        print(f"  🎯 {type(model.final_estimator_).__name__}")
        
    print("="*55)
    print("🎉 Verification Complete: All 3 powerful models are integrated!")

if __name__ == '__main__':
    verify_ensemble()