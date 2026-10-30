#!/usr/bin/env python
"""Benchmark the model against external predictors (AlphaMissense, REVEL, CADD, EVE, PrimateAI-3D, ...).
You must download their scores yourself (licences vary) into a CSV: variant_id,<score columns...>
variant_id format = protein_id:R248Q (same as tables/predictions_oof.csv).
   python tools/compare_external.py --pred results/tables/predictions_oof.csv --external scores.csv \
          --lower-is-pathogenic SIFT,PROVEAN --out results/figures/external_benchmark.png
Comparison is restricted to variants scored by ALL tools. Cluster-bootstrap 95% CI.
CAVEAT: many tools (REVEL, CADD, PrimateAI...) were trained on ClinVar-like labels -> circularity; state this in the paper."""
import argparse
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--external", required=True)
    ap.add_argument("--lower-is-pathogenic", default=""); ap.add_argument("--out", default="external_benchmark.png")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    p = pd.read_csv(a.pred); e = pd.read_csv(a.external)
    d = p.merge(e, on="variant_id", how="inner")
    flip = {c.strip() for c in a.lower_is_pathogenic.split(",") if c}
    tools = [c for c in e.columns if c != "variant_id"]
    for c in flip:
        d[c] = -d[c]
    d = d.dropna(subset=tools + ["proba"])
    print(f"{len(d)} variants scored by all tools ({len(d)/len(p):.1%} of predictions)", file=sys.stderr)
    scores = {"This model (OOF)": d["proba"].values, **{c: d[c].values for c in tools}}
    y, g = d["label"].values, d["cluster"].values
    idx = pd.Series(np.arange(len(d))).groupby(g).apply(lambda s: s.values).to_dict()
    ug = np.array(list(idx)); rng = np.random.default_rng(0)
    boot = {k: [] for k in scores}
    for _ in range(a.boot):
        ii = np.concatenate([idx[x] for x in rng.choice(ug, len(ug))])
        if y[ii].min() == y[ii].max():
            continue
        for k, s in scores.items():
            boot[k].append(roc_auc_score(y[ii], s[ii]))
    rows = [dict(tool=k, auroc=roc_auc_score(y, s), lo=np.percentile(boot[k], 2.5), hi=np.percentile(boot[k], 97.5),
                 auprc=average_precision_score(y, s)) for k, s in scores.items()]
    t = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    t.to_csv(Path(a.out).with_suffix(".csv"), index=False); print(t.round(3).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for k, s in scores.items():
        fpr, tpr, _ = roc_curve(y, s); ax[0].plot(fpr, tpr, lw=2.5 if "model" in k else 1.3, label=f"{k} ({roc_auc_score(y, s):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].legend(fontsize=8); ax[0].set(title="ROC on shared variants", xlabel="FPR", ylabel="TPR")
    ax[1].barh(t.tool, t.auroc, xerr=[t.auroc - t.lo, t.hi - t.auroc], color="#4c78a8", capsize=3)
    ax[1].set(title="AUROC (95% cluster-bootstrap CI)", xlim=(0.5, 1)); ax[1].invert_yaxis()
    plt.tight_layout(); plt.savefig(a.out, dpi=200)
