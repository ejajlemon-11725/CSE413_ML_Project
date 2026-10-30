# 🧬 Protein Pathogenicity Predictor & Advanced Validation Dashboard

Hybrid Machine Learning pipeline for predicting protein pathogenicity. Integrates ESM-2 (650M) embeddings and AlphaFold structural annotations with an advanced XGBoost/LGBM/CatBoost ensemble. Features a comprehensive Streamlit dashboard for real-time risk assessment, conformal prediction, and conformal validation on 62,727 genetic variants.

## 🌟 Features

- **🔬 Prediction Interface:** Predict variant pathogenicity via batch CSV upload (Top 300 Features) or interactive sequence inputs.
- **⚙️ Model Performance & Ablation:** View ROC curves, Precision-Recall curves, and the impact of ESM-2 embeddings.
- **🛡️ Safety & Calibration:** Clinical safety validation using Brier scores, reliability diagrams, and conformal risk-coverage trade-offs.
- **🧬 Protein Subgroup Analysis:** Homology-aware validation across specific protein families.
- **🌍 Feature Interpretation:** Confusion matrices and feature importance visualization.

## 📂 Repository Structure

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit dashboard application |
| `final_stacking_model.pkl` | Pre-trained stacking ensemble (XGBoost, LGBM, CatBoost) |
| `top_300_features.pkl` | List of the top 300 features required for prediction |
| `requirements.txt` | Required Python dependencies |
| `test_variants_100_mixed.csv` | Sample dataset for testing predictions |

## 🚀 Local Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/ejajlemon-11725/CSE413_ML_Project.git
   cd CSE413_ML_Project
   ```

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # Mac/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit dashboard**

   ```bash
   streamlit run app.py
   ```

The dashboard opens automatically at `http://localhost:8501`.

## ☁️ Streamlit Cloud Deployment

1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app**.
3. Fill in the deployment details:
   - **Repository:** `ejajlemon-11725/CSE413_ML_Project`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**.

Any future `git push` to `main` will automatically update the live app.

## 📊 How to Use the Dashboard

- **Method 1 (Batch Processing):** Go to the *Prediction Interface* tab and upload `test_variants_100_mixed.csv`. The dashboard outputs a classification table (Pathogenic vs Benign) for every variant.
- **Method 2 (Interactive Demo):** Paste a wild-type and a mutated protein sequence to see a simulated real-time gauge of the pathogenicity score.
