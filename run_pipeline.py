#!/usr/bin/env python
"""End-to-end pipeline: QC -> homology groups -> (locked hold-out) -> features -> repeated grouped CV ->
stacking -> statistics -> conformal -> explainability -> final model."""
import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mutpred import plots
from mutpred.config import Config
from mutpred.conformal import cal_scores, conformal_metrics
from mutpred.data import assign_clusters, load_dataset
from mutpred.esm import ESMFeatureBuilder, make_backend
from mutpred.evaluate import (auprc, auroc, cluster_bootstrap, conformal_table, error_analysis, full_metrics,
                              holdout_eval, learning_curve, per_protein_auc, pooled_table, run_cv, summarize,
                              summarize_numeric)
from mutpred.explain import contributions
from mutpred.features import build_blocks
from mutpred.extras import build_extra_blocks
from mutpred.models import BASELINE, STACK, default_specs, fit_model, proposed_name, stack


def parse():
    c = Config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=c.csv_path); ap.add_argument("--out", default=c.out_dir)
    ap.add_argument("--cache", default=c.cache_dir); ap.add_argument("--seed", type=int, default=c.seed)
    ap.add_argument("--esm-model", default=c.esm_model, help="e.g. facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--esm-backend", default=c.esm_backend, choices=["hf", "mock"])
    ap.add_argument("--no-esm", action="store_true")
    ap.add_argument("--batch-size", type=int, default=c.esm_batch_size)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--pca", type=int, default=c.pca_components)
    ap.add_argument("--splits", type=int, default=c.n_splits); ap.add_argument("--repeats", type=int, default=c.n_repeats)
    ap.add_argument("--threshold", default=c.threshold_method, choices=["inner", "fixed"])
    ap.add_argument("--tune", action="store_true"); ap.add_argument("--tune-iters", type=int, default=c.tune_iters)
    ap.add_argument("--bootstrap", type=int, default=c.n_bootstrap)
    ap.add_argument("--cluster", default=c.cluster_method, choices=["auto", "mmseqs", "kmer", "protein"])
    ap.add_argument("--holdout", type=float, default=0.0, help="fraction of homology clusters locked as final test set (e.g. 0.15)")
    ap.add_argument("--alpha", type=float, default=c.conformal_alpha, help="conformal miscoverage level")
    ap.add_argument("--learning-curve", action="store_true")
    ap.add_argument("--domains", default=None, help="optional CSV protein_id,start,end,name for track plots")
    ap.add_argument("--af-dir", default=None, help="folder with AlphaFold PDBs (tools/fetch_alphafold.py)")
    ap.add_argument("--msa-dir", default=None, help="folder with <protein_id>.a3m (tools/prepare_msa.py)")
    ap.add_argument("--annot", default=None, help="UniProt feature CSV (tools/fetch_uniprot_features.py)")
    ap.add_argument("--max-variants", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="1 repeat, 200 bootstrap")
    ap.add_argument("--dpi", type=int, default=c.dpi)
    a = ap.parse_args()
    cfg = Config(csv_path=a.csv, out_dir=a.out, cache_dir=a.cache, seed=a.seed, esm_model=a.esm_model,
                 esm_backend=a.esm_backend, use_esm=not a.no_esm, esm_batch_size=a.batch_size, esm_layer=a.layer,
                 pca_components=a.pca, n_splits=a.splits, n_repeats=1 if a.fast else a.repeats,
                 threshold_method=a.threshold, tune=a.tune, tune_iters=a.tune_iters,
                 n_bootstrap=200 if a.fast else a.bootstrap, cluster_method=a.cluster, holdout=a.holdout,
                 conformal_alpha=a.alpha, learning_curve=a.learning_curve, af_dir=a.af_dir, msa_dir=a.msa_dir, annot_csv=a.annot, max_variants=a.max_variants, dpi=a.dpi)
    return cfg, a.domains


def main():
    cfg, domains_path = parse()
    random.seed(cfg.seed); np.random.seed(cfg.seed)
    if cfg.threshold_method == "fixed":
        print("[warn] --threshold fixed disables calibration, stacking and conformal prediction")
    out = Path(cfg.out_dir); fig = out / "figures"; tab = out / "tables"
    for d in (fig, tab):
        d.mkdir(parents=True, exist_ok=True)
    json.dump(cfg.to_dict(), open(out / "config.json", "w"), indent=2)
    domains = pd.read_csv(domains_path) if domains_path else None

    # ---------------------------------------------------------------- 1. data, groups, locked hold-out
    df_all, qc = load_dataset(cfg)
    qc.to_csv(tab / "data_qc.csv", index=False)
    clusters, cl_method = assign_clusters(df_all, cfg)
    df_all["cluster"] = clusters
    plots.data_overview(df_all, clusters, fig / "01_data_overview.png", cfg.dpi)
    is_test = np.zeros(len(df_all), bool)
    if cfg.holdout > 0:
        rng = np.random.default_rng(cfg.seed)
        uc = np.unique(clusters)
        is_test = np.isin(clusters, rng.choice(uc, int(round(len(uc) * cfg.holdout)), replace=False))
        print(f"[holdout] {is_test.sum()} variants in {int(round(len(uc) * cfg.holdout))} locked clusters; "
              f"touched only once, at the very end")
    dev_idx, test_idx = np.where(~is_test)[0], np.where(is_test)[0]

    # ---------------------------------------------------------------- 2. features (PCA fitted on dev only)
    builder = ESMFeatureBuilder(make_backend(cfg), cfg.max_len, cfg.cache_dir) if cfg.use_esm else None
    blocks_all, reducer, lp_all, raw = build_blocks(df_all, builder, None, True, cfg.pca_components, fit_idx=dev_idx)
    extras = build_extra_blocks(df_all, cfg)
    blocks_all.update(extras)
    if extras:
        cov = {k: float(v.iloc[:, 0].mean()) for k, v in extras.items()}
        pd.Series(cov, name="fraction_of_variants_covered").to_csv(tab / "extra_feature_coverage.csv")
        print("[extras] coverage:", {k: round(v, 3) for k, v in cov.items()})
    df = df_all.iloc[dev_idx].reset_index(drop=True)
    blocks = {k: v.iloc[dev_idx].reset_index(drop=True) for k, v in blocks_all.items()}
    lp = lp_all[dev_idx] if lp_all is not None else None
    y, groups = df["label"].values, df["cluster"].values
    allfeat = pd.concat([blocks[k] for k in ("bio", "esm_scalar", "struct", "cons", "annot") if k in blocks], axis=1)
    pd.concat([df[["variant_id", "label", "cluster"]], allfeat], axis=1).to_csv(tab / "features_dev.csv", index=False)
    key = ["esm_llr", "esm_entropy", "esm_logp_wt", "grantham", "blosum62", "delta_hydropathy",
           "delta_volume", "ctx_hydropathy", "ctx_complexity", "rel_pos", "esm_mut_rank", "delta_charge",
           "af_plddt", "af_cn10", "af_hse_ratio", "cons_entropy", "cons_pssm_delta", "cons_llr", "ann_dist_func_site"]
    plots.feature_distributions(allfeat, y, key, fig / "02_feature_distributions.png", cfg.dpi)
    plots.correlation(allfeat.drop(columns=[c for c in allfeat if c.endswith("_available")]), fig / "03_feature_correlation.png", cfg.dpi)
    plots.risk_curves(allfeat, y, key + ["grantham", "blosum62", "esm_logp_mut", "af_plddt_win", "cons_entropy_win",
                                         "cons_pssm_mut", "af_cb_cn8", "ann_n_overlap"], fig / "03b_risk_curves.png", dpi=cfg.dpi)

    # ---------------------------------------------------------------- 3. repeated, homology-grouped, nested CV
    specs = default_specs(cfg.use_esm, tuple(extras))
    if cfg.threshold_method == "fixed":
        specs = [s for s in specs if s.kind != "stack"]
    ref = BASELINE
    prop = proposed_name(specs)
    res = run_cv(blocks, y, groups, specs, cfg)
    order = [s.name for s in specs]
    summ = summarize(res.fold_df, order); summ.to_csv(tab / "cv_metrics_mean_sd.csv")
    num = summarize_numeric(res.fold_df, order)
    pooled = pooled_table(res); pooled.to_csv(tab / "pooled_oof_metrics.csv")
    res.fold_df.to_csv(tab / "cv_metrics_per_fold.csv", index=False)
    ct = conformal_table(res.fold_df); ct.to_csv(tab / "conformal_summary.csv")
    print("\n" + summ[["auroc", "auprc", "mcc", "bal_acc", "brier"]].to_string() + "\n")

    scores = {s.name: res.pooled(s.name) for s in specs if not np.isnan(res.pooled(s.name)).all()}
    boot = cluster_bootstrap(y, scores, groups, cfg.n_bootstrap, ref, cfg.seed)
    boot.to_csv(tab / "bootstrap_ci_and_paired_tests.csv")

    # ---------------------------------------------------------------- 4. performance figures
    trained = [n for n in order if n in scores and not np.isnan(res.oof_cal[n]).all()]
    best = max(trained, key=lambda n: pooled.loc[n, "auroc"])
    show = [n for n in [BASELINE, "Zero-shot: ESM LLR", prop, STACK] if n in scores]
    plots.roc_pr(res, show, fig / "04_roc_pr.png", cfg.dpi)
    plots.ablation(boot, ref, fig / "05_ablation.png", cfg.dpi)
    plots.metrics_heatmap(num, fig / "06_metrics_heatmap.png", cfg.dpi)
    plots.fold_stability(res.fold_df, order, fig / "07_fold_stability.png", cfg.dpi)
    cal_models = {n: res.pooled(n, True) for n in show if n in trained}
    plots.calibration(y, cal_models, fig / "08_calibration.png", cfg.dpi)
    plots.decision_curve(y, cal_models, fig / "09_decision_curve.png", cfg.dpi)
    thr_of = {n: float(res.fold_df.query("model == @n")["thr"].mean()) for n in trained}
    plots.confusion_grid(y, {n: (res.pooled(n, True), thr_of[n]) for n in trained}, fig / "10_confusion_all_models.png", cfg.dpi)
    plots.error_overlap(y, {n: (res.pooled(n, True), thr_of[n]) for n in trained}, fig / "10b_error_overlap.png", cfg.dpi)
    plots.risk_coverage(y, {n: (res.pooled(n, True), thr_of[n]) for n in show if n in trained}, fig / "10c_risk_coverage.png", cfg.dpi)
    plots.confusion_thresholds(y, res.pooled(prop, True), thr_of[prop], df["label_strength"].values, prop,
                               fig / "11_confusion_operating_points.png", cfg.dpi)
    plots.confusion_folds(res, prop, fig / "12_confusion_fold_variability.png", cfg.dpi)
    plots.operating_point(y, res.pooled(prop, True), thr_of[prop], prop, fig / "13_operating_point.png", cfg.dpi)
    pp = per_protein_auc(df, scores[prop]); pp.to_csv(tab / "per_protein_auc.csv", index=False)
    if len(pp):
        plots.per_protein(pp, fig / "14_per_protein_auc.png", cfg.dpi)
    plots.subgroup_performance(df, allfeat, scores[prop], fig / "15_subgroup_performance.png", cfg.dpi)
    if len(res.stack_w):
        plots.stack_weights(res.stack_w, fig / "16_stack_weights.png", cfg.dpi)
        res.stack_w.groupby("base")["weight"].agg(["mean", "std"]).to_csv(tab / "stack_weights.csv")

    # conformal
    for n in dict.fromkeys([prop, best]):
        pv = res.pval[n][0]
        if not np.isnan(pv).all():
            plots.conformal_panel(y, pv, n, cfg.conformal_alpha,
                                  fig / f"17_conformal_{'proposed' if n == prop else 'best'}.png", cfg.dpi)

    # label-strength sensitivity analysis
    rows = []
    for lab in ["definite", "likely"]:
        m = (df.label_strength == lab).values
        if m.sum() > 20 and len(np.unique(y[m])) == 2:
            rows += [dict(subset=lab, model=n, n=int(m.sum()), auroc=auroc(y[m], scores[n][m]),
                          auprc=auprc(y[m], scores[n][m])) for n in show]
    pd.DataFrame(rows).to_csv(tab / "sensitivity_label_strength.csv", index=False)

    # ---------------------------------------------------------------- 5. explainability (out-of-fold TreeSHAP)
    c = res.contrib.get(prop)
    d = None
    if c is not None:
        V, X, names = c["values"], c["X"], c["names"]
        pc = res.pooled(prop, True); thr = thr_of[prop]
        plots.shap_beeswarm(V, X, names, fig / "20_shap_beeswarm.png", dpi=cfg.dpi)
        imp = plots.shap_importance(V, names, fig / "21_shap_importance.png", dpi=cfg.dpi)
        imp.to_csv(tab / "shap_importance.csv", index=False)
        plots.shap_dependence(V, X, names, y, fig / "22_shap_dependence.png", dpi=cfg.dpi)
        d, etabs = error_analysis(df, pc, thr)
        d.to_csv(tab / "predictions_oof.csv", index=False)
        for k, t in etabs.items():
            t.to_csv(tab / f"errors_{k}.csv")
        outcome = np.where((d.pred == 1) & (d.label == 1), "TP", np.where((d.pred == 0) & (d.label == 0), "TN",
                           np.where(d.pred == 1, "FP", "FN")))
        plots.shap_by_outcome(V, names, outcome, fig / "23_shap_by_outcome.png", dpi=cfg.dpi)
        plots.error_heatmaps(df.join(allfeat[["esm_llr"]]) if "esm_llr" in allfeat else df, d,
                             fig / "24_error_analysis.png", cfg.dpi)
        cs = fig / "case_studies"; cs.mkdir(exist_ok=True)
        pick = {"true_pathogenic": d[(d.error == "correct") & (d.label == 1)].sort_values("proba", ascending=False),
                "true_benign": d[(d.error == "correct") & (d.label == 0)].sort_values("proba"),
                "false_positive": d[d.error == "FP"].sort_values("proba", ascending=False),
                "false_negative": d[d.error == "FN"].sort_values("proba")}
        for kind, t in pick.items():
            for i in t.head(cfg.n_case_studies).index:
                plots.variant_explanation(V[i], c["base"][i], X[i], names,
                                          f"{kind}: {d.variant_id[i]} (P={d.proba[i]:.2f}, true={d.label[i]})",
                                          cs / f"{kind}_{d.variant_id[i].replace(':', '_')}.png", dpi=cfg.dpi)
        if "esm_llr" in allfeat:
            top = df.groupby("protein_key").agg(n=("label", "size"), k=("label", "nunique"))
            top = top[top.k == 2].sort_values("n", ascending=False)
            for pk in top.index[:3]:
                plots.protein_track(df, allfeat["esm_llr"].values, d["proba"].values, thr, pk,
                                    fig / f"25_protein_track_{str(pk).replace('/', '_')}.png", domains, cfg.dpi)
    if cfg.use_esm:
        plots.embedding_space(blocks["esm_pca"], y, (d.error == "correct").values if d is not None else None,
                              fig / "26_embedding_space.png", cfg.dpi)
        bp = df.groupby("protein_key").agg(n=("label", "size"), k=("label", "nunique"))
        bp = bp[bp.k == 2].sort_values("n", ascending=False)
        if len(bp):
            plots.landscape(df, lp, bp.index[0], fig / "27_esm_landscape_top_protein.png", cfg.dpi)

    if cfg.learning_curve:
        lc = learning_curve(blocks, y, groups, [s for s in specs if s.name in (BASELINE, prop)], cfg)
        lc.to_csv(tab / "learning_curve.csv", index=False)
        plots.learning_curve_plot(lc, fig / "28_learning_curve.png", cfg.dpi)

    # ---------------------------------------------------------------- 6. LOCKED HOLD-OUT (evaluated once)
    ho_md = ""
    if len(test_idx):
        yall, gall = df_all["label"].values, df_all["cluster"].values
        o = holdout_eval(blocks_all, yall, gall, specs, dev_idx, test_idx, cfg)
        yt = yall[test_idx]
        rows, sc_t, pr_t = [], {}, {}
        for n, r in o.items():
            thr_ = None if r["fit"] is None else r["fit"].threshold
            rows.append(dict(model=n, **full_metrics(yt, r["p"], r["pc"], thr_)))
            sc_t[n] = r["p"]
            if r["pc"] is not None:
                pr_t[n] = (r["pc"], thr_)
        hm = pd.DataFrame(rows).set_index("model").loc[[n for n in order if n in o]]
        hm.to_csv(tab / "holdout_metrics.csv")
        hb = cluster_bootstrap(yt, sc_t, gall[test_idx], cfg.n_bootstrap, ref, cfg.seed)
        hb.to_csv(tab / "holdout_bootstrap.csv")
        plots.confusion_grid(yt, pr_t, fig / "30_holdout_confusion_all_models.png", cfg.dpi,
                             title="LOCKED HOLD-OUT test set — confusion matrices")
        plots.metrics_heatmap(hm[[c_ for c_ in summarize_numeric(res.fold_df, order).columns]], fig / "31_holdout_metrics_heatmap.png", cfg.dpi)
        if prop in o and "pv" in o[prop]:
            m = conformal_metrics(yt, o[prop]["pv"], cfg.conformal_alpha)
            pd.Series(m).to_csv(tab / "holdout_conformal.csv")
            plots.conformal_panel(yt, o[prop]["pv"], prop + " (hold-out)", cfg.conformal_alpha,
                                  fig / "32_holdout_conformal.png", cfg.dpi)
        ho_md = (f"\n## Locked hold-out ({len(test_idx)} variants, never used for development)\n\n"
                 f"{hm[['auroc','auprc','mcc','bal_acc','sensitivity','specificity','ppv','brier']].round(3).to_markdown()}\n\n"
                 f"Bootstrap (cluster) CI / paired tests:\n\n{hb.round(4).to_markdown()}\n")

    # ---------------------------------------------------------------- 7. final model on ALL data (for predict.py)
    spec = next(s for s in specs if s.name == prop)
    Xa, names_a = stack(blocks_all, spec)
    fit = fit_model(spec, Xa, df_all["label"].values, df_all["cluster"].values, cfg, cfg.seed)
    cs_ = cal_scores(fit.oof_cal, df_all["label"].values) if fit.oof_cal is not None else None
    joblib.dump(dict(fit=fit, names=names_a, spec=spec, reducer=reducer, cfg=cfg.to_dict(),
                     bio_cols=list(blocks_all["bio"].columns), train_X_median=np.nanmedian(Xa, 0),
                     conformal_scores=cs_), out / "final_model.joblib")

    with open(out / "REPORT.md", "w") as f:
        f.write(f"# Results\n\nVariants: {len(df_all)} (dev {len(dev_idx)}, locked test {len(test_idx)}) | "
                f"proteins: {df_all.protein_key.nunique()} | homology clusters: {len(set(clusters))} ({cl_method}) | "
                f"CV: {cfg.n_repeats}x{cfg.n_splits} StratifiedGroupKFold by cluster, nested calibration/threshold\n\n"
                f"Best model by pooled AUROC: **{best}**\n\n"
                f"## Cross-validated metrics (mean ± SD over folds)\n\n{summ.to_markdown()}\n\n"
                f"## Pooled out-of-fold, cluster-bootstrap CI and paired tests vs `{ref}` (Holm-adjusted)\n\n"
                f"{boot.round(4).to_markdown()}\n\n"
                f"## Conformal prediction (alpha={cfg.conformal_alpha}; per-fold mean)\n\n{ct.round(3).to_markdown()}\n"
                f"{ho_md}\n## Data QC\n\n{qc.to_markdown(index=False)}\n")
    print(f"\nDone. See {out}/REPORT.md, {fig}/ and {tab}/")


if __name__ == "__main__":
    main()
