# Protein missense variant effect prediction (ESM-2 + biochemistry + stacking + conformal + explainability)

Input: `clinical_substitutions.csv` with `protein_id, protein_sequence, mutant, mutated_sequence, annotation`.

```bash
pip install -r requirements.txt
python tests/test_core.py                                              # unit tests (7)
python run_pipeline.py --esm-backend mock --fast --splits 3            # 1-minute smoke test (meaningless numbers)

# recommended paper run (GPU strongly advised for ~62k variants)
python run_pipeline.py --csv clinical_substitutions.csv --holdout 0.15 --learning-curve \
       --esm-model facebook/esm2_t33_650M_UR50D --batch-size 8
python run_pipeline.py --tune                                          # in-fold random-search tuning
python tools/compare_runs.py results_35M results_650M                  # scaling comparison
python predict.py --bundle results/final_model.joblib --sequence MKT... --mutant R248Q --plot why.png
```
First run caches ESM outputs in `cache/`; re-runs are fast. Use `--cluster mmseqs` with MMseqs2 installed.

## Optional evidence sources (largest expected gains)
```bash
python tools/fetch_alphafold.py        --csv clinical_substitutions.csv --out af_models          # needs internet
python tools/fetch_uniprot_features.py --csv clinical_substitutions.csv --out annotations.csv     # needs internet
python tools/prepare_msa.py --csv clinical_substitutions.csv --out msa_work                       # then run colabfold_search (see docstring)
python run_pipeline.py --holdout 0.15 --af-dir af_models --msa-dir msa_work/msa --annot annotations.csv
python tools/compare_external.py --pred results/tables/predictions_oof.csv --external scores.csv   # AlphaMissense/REVEL/...
```
`protein_id` must be a **UniProt accession** for the structure/annotation tools. Missing files give NaN features
(`*_available` flags); coverage is written to `tables/extra_feature_coverage.csv`.
The ablation ladder then adds one source at a time (Bio+ESM -> +AlphaFold -> +MSA -> +UniProt -> ALL).

## Pipeline
1. QC (`tables/data_qc.csv`) -> homology clusters -> optional **locked hold-out** of whole clusters (evaluated once, last).
2. Features: biochemical/context (27) + ESM-2 masked-marginal (7) + ESM embeddings -> PCA (fit on dev only).
3. **Repeated StratifiedGroupKFold** by cluster; inside each training fold: inner grouped CV -> Platt calibration,
   MCC threshold, stacking meta-learner and conformal calibration scores (test fold never used for any decision).
4. 12 models: zero-shot (BLOSUM, Grantham, ESM LLR), LR, RF, GBM on different feature sets, **stack (meta-LR)**.
5. Statistics: cluster-bootstrap CI, paired bootstrap tests vs baseline with Holm correction, fold-level SD.
6. **Class-conditional conformal prediction** (pathogenic / benign / uncertain) with coverage validation.
7. Explainability: out-of-fold TreeSHAP, SHAP by outcome (TP/FP/FN/TN), case studies, error analysis.

## Outputs (`results/`)
`REPORT.md`, `tables/*.csv`, `figures/*.png` (see figure index below), `final_model.joblib`.

| Figures | |
|---|---|
| 01-03 | data overview, feature distributions, correlation |
| 04-09 | ROC/PR, ablation+CI, metrics heatmap, fold stability, calibration, decision curve |
| 10-13 | confusion: all models, operating points + label confidence, fold variability, threshold analysis |
| 03b, 10b, 10c | risk curves per feature, error overlap between models, risk-coverage (selective prediction) |
| 14-16 | per-protein AUROC, subgroup performance, stacking weights |
| 17 | conformal prediction (coverage, set composition, confident accuracy) |
| 20-24 | SHAP beeswarm / importance / dependence / by outcome, error heatmaps |
| 25-28 | protein tracks (+optional domains), embedding space, ESM landscape, learning curve |
| 30-32 | locked hold-out: confusion, metrics, conformal |

## Limitations
- Structure/MSA/annotation code is tested only on synthetic files; the download tools need internet and were not run.
- Secondary structure is a coarse Ramachandran call (no DSSP); SASA is approximated by contact numbers / half-sphere exposure.
- Without xgboost: HistGradientBoosting + occlusion attribution (not exact SHAP).
- Stack/conformal need `--threshold inner` (default).
- Real ESM-2 (`--esm-backend hf`) and XGBoost paths were not executable in the authoring sandbox.
