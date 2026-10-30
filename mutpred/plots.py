"""All figures. Every function saves a PNG and closes the figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .constants import AA, AA_INDEX, CLASS_ORDER
from .features import feature_block_of

sns.set_theme(style="whitegrid", font_scale=1.0)
C_PATH, C_BEN = "#d95f02", "#1b9e77"
BLOCK_COLORS = {"Biochemical substitution": "#4c78a8", "ESM masked-marginal": "#e45756",
                "ESM embeddings (PCA)": "#f2a65a", "Sequence context": "#72b7b2",
                "AlphaFold structure": "#59a14f", "Conservation (MSA)": "#b07aa1", "UniProt annotation": "#9c755f"}


def _save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def data_overview(df, clusters, path, dpi=200):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    vc = df["label"].map({0: "Benign", 1: "Pathogenic"}).value_counts()
    sns.barplot(x=vc.index, y=vc.values, hue=vc.index, palette={"Benign": C_BEN, "Pathogenic": C_PATH},
                legend=False, ax=ax[0, 0])
    for i, v in enumerate(vc.values):
        ax[0, 0].text(i, v, str(v), ha="center", va="bottom")
    ax[0, 0].set_title("A. Class balance")
    cnt = df["protein_key"].value_counts()
    ax[0, 1].hist(cnt.values, bins=np.unique(np.logspace(0, np.log10(max(cnt.max(), 2)), 25).astype(int)),
                  color="#4c78a8")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_title("B. Variants per protein"); ax[0, 1].set_xlabel("# variants")
    g = df.groupby("protein_key")["label"].agg(["mean", "size"])
    ax[0, 2].hist(g[g["size"] >= 3]["mean"], bins=20, color="#b279a2")
    ax[0, 2].set_title("C. Pathogenic fraction per protein (n>=3)\n(U-shape = protein-level label bias)")
    ax[0, 2].set_xlabel("fraction pathogenic")
    for lab, col, nm in [(0, C_BEN, "Benign"), (1, C_PATH, "Pathogenic")]:
        sns.kdeplot(df[df.label == lab]["rel_pos"], ax=ax[1, 0], color=col, label=nm, fill=True, alpha=.3,
                    common_norm=False, clip=(0, 1))
    ax[1, 0].set_title("D. Relative position in protein"); ax[1, 0].legend()
    M = np.full((20, 20), np.nan); N = np.zeros((20, 20))
    for w, m, l in zip(df.wt_aa, df.mut_aa, df.label):
        i, j = AA_INDEX[w], AA_INDEX[m]
        N[i, j] += 1; M[i, j] = (0 if np.isnan(M[i, j]) else M[i, j]) + l
    F = np.where(N >= 3, M / np.maximum(N, 1), np.nan)
    sns.heatmap(F, ax=ax[1, 1], xticklabels=list(AA), yticklabels=list(AA), cmap="RdYlBu_r", vmin=0, vmax=1,
                cbar_kws={"label": "fraction pathogenic"})
    ax[1, 1].set_title("E. Pathogenic fraction by substitution (n>=3)"); ax[1, 1].set_xlabel("mutant"); ax[1, 1].set_ylabel("wild-type")
    cs = pd.Series(clusters).value_counts()
    ax[1, 2].hist(df.drop_duplicates("protein_key").assign(c=pd.Series(clusters, index=df.index))
                  .groupby("c").size(), bins=20, color="#59a14f")
    ax[1, 2].set_yscale("log"); ax[1, 2].set_title(f"F. Proteins per homology cluster ({len(cs)} clusters)")
    _save(fig, path, dpi)


def feature_distributions(feat, y, cols, path, dpi=200):
    cols = [c for c in cols if c in feat.columns]
    nc = 4; nr = int(np.ceil(len(cols) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr))
    for a, c in zip(np.atleast_1d(ax).ravel(), cols):
        for lab, col, nm in [(0, C_BEN, "Benign"), (1, C_PATH, "Pathogenic")]:
            v = feat[c].values[y == lab]
            a.hist(v, bins=30, density=True, histtype="stepfilled", alpha=.45, color=col, label=nm)
        a.set_title(c)
    for a in np.atleast_1d(ax).ravel()[len(cols):]:
        a.axis("off")
    np.atleast_1d(ax).ravel()[0].legend()
    _save(fig, path, dpi)


def correlation(feat, path, dpi=200):
    f = feat.loc[:, feat.std() > 0]
    c = f.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.38 * len(c) + 4, 0.34 * len(c) + 3))
    sns.heatmap(c, mask=np.triu(np.ones_like(c, bool), 1), cmap="vlag", center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "Spearman"})
    ax.set_title("Feature correlation (redundancy check)")
    _save(fig, path, dpi)


def _fold_curves(res, name, kind):
    grid = np.linspace(0, 1, 101)
    curves = []
    R = res.oof_raw[name].shape[0]
    for r in range(R):
        for f in np.unique(res.fold_id[r]):
            idx = np.where(res.fold_id[r] == f)[0]
            s = res.oof_raw[name][r, idx]; yy = res.y[idx]
            if len(np.unique(yy)) < 2 or np.isnan(s).any():
                continue
            if kind == "roc":
                fpr, tpr, _ = roc_curve(yy, s); curves.append(np.interp(grid, fpr, tpr))
            else:
                p, rc, _ = precision_recall_curve(yy, s)
                curves.append(np.interp(grid, rc[::-1], p[::-1]))
    return grid, np.array(curves)


def roc_pr(res, names, path, dpi=200):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    pal = sns.color_palette("tab10", len(names))
    for c, n in zip(pal, names):
        for k, kind in enumerate(["roc", "pr"]):
            g, cv = _fold_curves(res, n, kind)
            m, s = cv.mean(0), cv.std(0)
            from .evaluate import auroc, auprc
            val = (auroc if kind == "roc" else auprc)(res.y, res.pooled(n))
            ax[k].plot(g, m, color=c, lw=2, label=f"{n} ({val:.3f})")
            ax[k].fill_between(g, m - s, m + s, color=c, alpha=.12)
    ax[0].plot([0, 1], [0, 1], "k--", alpha=.4)
    ax[1].axhline(res.y.mean(), color="k", ls="--", alpha=.4)
    ax[0].set(title="A. ROC (mean ± SD over grouped folds; legend = pooled AUROC)", xlabel="1 - specificity", ylabel="sensitivity")
    ax[1].set(title="B. Precision-recall (legend = pooled AUPRC)", xlabel="recall", ylabel="precision")
    ax[0].legend(loc="lower right", fontsize=8); ax[1].legend(loc="upper right", fontsize=8)
    _save(fig, path, dpi)


def ablation(boot, ref, path, dpi=200):
    fig, ax = plt.subplots(1, 2, figsize=(15, 0.5 * len(boot) + 3), sharey=True)
    order = boot.sort_values("auroc").index
    for a, met in zip(ax, ["auroc", "auprc"]):
        b = boot.loc[order]
        cols = ["#e45756" if "proposed" in i else ("#999999" if i.startswith("Zero") else "#4c78a8") for i in b.index]
        a.barh(range(len(b)), b[met], color=cols,
               xerr=[b[met] - b[f"{met}_lo"], b[f"{met}_hi"] - b[met]], capsize=3)
        for i, (idx, r) in enumerate(b.iterrows()):
            p = r[f"{met}_p_vs_ref"]
            if idx != ref:
                a.text(r[f"{met}_hi"] + .005, i, f"Δ={r[f'{met}_diff_vs_ref']:+.3f} (p={p:.3f})", va="center", fontsize=7)
        a.set_yticks(range(len(b))); a.set_yticklabels(b.index)
        a.set_xlim(max(0.3, b[f"{met}_lo"].min() - .05), min(1.12, b[f"{met}_hi"].max() + .2))
        a.set_title(f"{met.upper()} (pooled OOF, 95% homology-cluster bootstrap CI)\nΔ/p vs. {ref}")
    _save(fig, path, dpi)


def calibration(y, curves, path, dpi=200):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    pal = sns.color_palette("tab10", len(curves))
    for c, (n, p) in zip(pal, curves.items()):
        fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax[0].plot(mp, fr, "o-", color=c, label=n)
        ax[1].hist(p, bins=30, alpha=.4, color=c, label=n)
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set(title="A. Reliability (calibrated, out-of-fold)", xlabel="predicted P(pathogenic)", ylabel="observed fraction")
    ax[1].set(title="B. Distribution of predicted probabilities", xlabel="predicted P(pathogenic)")
    ax[0].legend(fontsize=8)
    _save(fig, path, dpi)


def operating_point(y, p, thr, name, path, dpi=200):
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
    ts = np.linspace(0.01, 0.99, 99)
    sens = [((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) for t in ts]
    spec = [((p < t) & (y == 0)).sum() / max((y == 0).sum(), 1) for t in ts]
    ppv = [((p >= t) & (y == 1)).sum() / max((p >= t).sum(), 1) for t in ts]
    from sklearn.metrics import matthews_corrcoef
    mcc = [matthews_corrcoef(y, (p >= t).astype(int)) for t in ts]
    for v, n in [(sens, "sensitivity"), (spec, "specificity"), (ppv, "PPV"), (mcc, "MCC")]:
        ax[0].plot(ts, v, label=n)
    ax[0].axvline(thr, color="k", ls="--", label=f"chosen thr={thr:.2f}")
    ax[0].set(title="A. Threshold analysis", xlabel="threshold"); ax[0].legend()
    for lab, col, nm in [(0, C_BEN, "Benign"), (1, C_PATH, "Pathogenic")]:
        sns.kdeplot(p[y == lab], ax=ax[1], color=col, fill=True, alpha=.35, label=nm, clip=(0, 1), common_norm=False)
    ax[1].axvline(thr, color="k", ls="--"); ax[1].set(title="B. Score separation", xlabel="P(pathogenic)"); ax[1].legend()
    cm = confusion_matrix(y, (p >= thr).astype(int))
    pct = cm / cm.sum(1, keepdims=True)
    lab = np.array([[f"{c}\n({q:.1%})" for c, q in zip(r1, r2)] for r1, r2 in zip(cm, pct)])
    sns.heatmap(cm, annot=lab, fmt="", cmap="Blues", cbar=False, ax=ax[2],
                xticklabels=["Benign", "Pathogenic"], yticklabels=["Benign", "Pathogenic"])
    ax[2].set(title="C. Confusion matrix (row-normalised %)", xlabel="predicted", ylabel="true")
    fig.suptitle(name)
    _save(fig, path, dpi)


def per_protein(pp, path, dpi=200):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(pp["auroc"], bins=20, color="#4c78a8")
    ax[0].axvline(pp["auroc"].median(), color="r", ls="--", label=f"median {pp['auroc'].median():.2f}")
    ax[0].set(title=f"A. Per-protein AUROC ({len(pp)} proteins with both classes)", xlabel="AUROC"); ax[0].legend()
    sc = ax[1].scatter(pp["n"], pp["auroc"], c=pp["frac_path"], cmap="RdYlBu_r", s=35, edgecolor="k", lw=.3)
    ax[1].set_xscale("log"); ax[1].set(title="B. AUROC vs. #variants", xlabel="# variants", ylabel="AUROC")
    plt.colorbar(sc, ax=ax[1], label="fraction pathogenic")
    _save(fig, path, dpi)


# ----------------------------------------------------------- explainability
def _swarm(vals, width=0.4, nb=40):
    off = np.zeros(len(vals))
    if np.ptp(vals) == 0:
        return off
    bins = np.digitize(vals, np.linspace(vals.min(), vals.max() + 1e-9, nb))
    rng = np.random.default_rng(0)
    for b in np.unique(bins):
        ix = np.where(bins == b)[0]
        if len(ix) > 1:
            off[ix] = (np.arange(len(ix)) - (len(ix) - 1) / 2)[rng.permutation(len(ix))]
    return off * width / max(np.abs(off).max(), 1)


def shap_beeswarm(values, X, names, path, top=20, dpi=200):
    ok = ~np.isnan(values).any(1)
    V, Xv = values[ok], X[ok]
    if len(V) > 8000:
        sel = np.random.default_rng(0).choice(len(V), 8000, replace=False)
        V, Xv = V[sel], Xv[sel]
    imp = np.abs(V).mean(0)
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(order) + 2))
    for row, j in enumerate(order):
        xv = Xv[:, j]
        col = rankdata(np.nan_to_num(xv)) / len(xv) if np.ptp(np.nan_to_num(xv)) > 0 else np.full(len(xv), .5)
        sc = ax.scatter(V[:, j], row + _swarm(V[:, j]), c=col, cmap="coolwarm", s=6, vmin=0, vmax=1, alpha=.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[j] for j in order])
    ax.axvline(0, color="k", lw=.6)
    ax.set(xlabel="SHAP value (log-odds; → more pathogenic)", title="Global explanation (out-of-fold SHAP)")
    cb = plt.colorbar(sc, ax=ax, pad=.01); cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"]); cb.set_label("feature value")
    _save(fig, path, dpi)


def shap_importance(values, names, path, top=20, dpi=200):
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.nanmean(np.abs(values), 0)})
    imp["block"] = imp.feature.map(feature_block_of)
    fig, ax = plt.subplots(1, 2, figsize=(15, 0.33 * top + 2.5), gridspec_kw={"width_ratios": [2, 1]})
    t = imp.sort_values("mean_abs_shap", ascending=False).head(top)
    ax[0].barh(t.feature[::-1], t.mean_abs_shap[::-1], color=[BLOCK_COLORS[b] for b in t.block[::-1]])
    ax[0].set(title=f"A. Top {top} features (mean |SHAP|)", xlabel="mean |SHAP|")
    b = imp.groupby("block")["mean_abs_shap"].sum().sort_values()
    ax[1].barh(b.index, b.values, color=[BLOCK_COLORS[i] for i in b.index])
    ax[1].set(title="B. Contribution by feature group\n(sum of mean |SHAP|)", xlabel="Σ mean |SHAP|")
    h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in BLOCK_COLORS.values()]
    ax[0].legend(h, BLOCK_COLORS.keys(), loc="lower right", fontsize=8)
    _save(fig, path, dpi)
    return imp.sort_values("mean_abs_shap", ascending=False)


def shap_dependence(values, X, names, y, path, k=6, dpi=200):
    ok = ~np.isnan(values).any(1)
    if ok.sum() > 8000:
        keep = np.where(ok)[0]
        ok = np.zeros(len(ok), bool)
        ok[np.random.default_rng(0).choice(keep, 8000, replace=False)] = True
    order = np.argsort(np.abs(values[ok]).mean(0))[::-1][:k]
    nc = 3; nr = int(np.ceil(k / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr))
    for a, j in zip(np.atleast_1d(ax).ravel(), order):
        x, s, yy = X[ok][:, j], values[ok][:, j], y[ok]
        a.scatter(x, s, c=np.where(yy == 1, C_PATH, C_BEN), s=6, alpha=.5)
        if np.ptp(x) > 0 and len(np.unique(x)) > 8:
            q = pd.qcut(x, 15, duplicates="drop")
            m = pd.DataFrame({"x": x, "s": s}).groupby(q, observed=True).median()
            a.plot(m.x, m.s, "k-", lw=2)
        a.axhline(0, color="gray", lw=.6); a.set(xlabel=names[j], ylabel="SHAP", title=names[j])
    for a in np.atleast_1d(ax).ravel()[len(order):]:
        a.axis("off")
    _save(fig, path, dpi)


def variant_explanation(vals, base, x, names, title, path, top=12, dpi=200):
    o = np.argsort(np.abs(vals))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(8, 0.4 * top + 1.8))
    ax.barh(range(len(o)), vals[o], color=[C_PATH if v > 0 else C_BEN for v in vals[o]])
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([f"{names[j]} = {x[j]:.2f}" for j in o], fontsize=8)
    ax.axvline(0, color="k", lw=.6)
    ax.set(xlabel="contribution to log-odds", title=f"{title}\nbase={base:.2f}  →  margin={base + vals.sum():.2f}")
    _save(fig, path, dpi)


def embedding_space(frame, y, correct, path, dpi=200):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    for a, (c1, c2, t) in zip(ax, [("ESMdelta_PC1", "ESMdelta_PC2", "Δ-embedding (mutant − WT)"),
                                  ("ESMwt_PC1", "ESMwt_PC2", "WT residue context")]):
        if c1 not in frame:
            a.axis("off"); continue
        a.scatter(frame[c1], frame[c2], c=np.where(y == 1, C_PATH, C_BEN), s=7, alpha=.5)
        if correct is not None:
            w = ~correct
            a.scatter(frame[c1][w], frame[c2][w], facecolors="none", edgecolors="k", s=18, lw=.4)
        a.set(title=f"{t} — PCA (orange=pathogenic, green=benign; ring=misclassified)", xlabel=c1, ylabel=c2)
        a.title.set_fontsize(9)
    _save(fig, path, dpi)


def error_heatmaps(df, d, path, dpi=200):
    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    for a, (col, cm, t) in zip(ax[:2], [("esm_llr", "RdBu", "Mean ESM LLR"), ("is_error", "Reds", "Error rate")]):
        v = df[col] if col in df else d[col]
        M = np.full((20, 20), np.nan)
        g = pd.DataFrame({"w": df.wt_aa, "m": df.mut_aa, "v": np.asarray(v)}).groupby(["w", "m"])["v"].agg(["mean", "size"])
        for (w, m), r in g.iterrows():
            if r["size"] >= 3:
                M[AA_INDEX[w], AA_INDEX[m]] = r["mean"]
        sns.heatmap(M, ax=a, xticklabels=list(AA), yticklabels=list(AA), cmap=cm,
                    center=0 if col == "esm_llr" else None)
        a.set(title=f"{t} by substitution (n>=3)", xlabel="mutant", ylabel="wild-type")
    T = d.groupby(["wt_class", "mut_class"])["is_error"].mean().unstack().reindex(index=CLASS_ORDER, columns=CLASS_ORDER)
    sns.heatmap(T, annot=True, fmt=".2f", cmap="Reds", ax=ax[2])
    ax[2].set(title="Error rate by residue-class transition", xlabel="mutant class", ylabel="wild-type class")
    _save(fig, path, dpi)


def landscape(df, lp, pk_label, path, dpi=200):
    """ESM predicted log-prob change (vs WT) for every substitution at observed sites of one protein."""
    sub = df[df.protein_key == pk_label]
    sites = sorted(sub.pos.unique())
    M = np.zeros((20, len(sites)))
    for j, p in enumerate(sites):
        i = sub.index[sub.pos == p][0]
        M[:, j] = lp[i] - lp[i, AA_INDEX[sub.loc[i, "wt_aa"]]]
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(sites) + 3), 7))
    sns.heatmap(M, ax=ax, cmap="RdBu", center=0, yticklabels=list(AA), cbar_kws={"label": "log P(aa) − log P(WT)"})
    ax.set_xticks(np.arange(len(sites)) + .5)
    ax.set_xticklabels([f"{sub[sub.pos == p].wt_aa.iloc[0]}{p}" for p in sites], rotation=90, fontsize=7)
    for _, r in sub.iterrows():
        ax.scatter(sites.index(r.pos) + .5, AA_INDEX[r.mut_aa] + .5, s=90, facecolors="none",
                   edgecolors=C_PATH if r.label == 1 else "k", lw=2 if r.label == 1 else 1.2,
                   marker="o" if r.label == 1 else "s")
    ax.set_title(f"ESM mutational landscape — {pk_label}\n(circle = pathogenic, square = benign observed variant)")
    _save(fig, path, dpi)


# =====================================================================  v2 additions
import textwrap
from sklearn.metrics import matthews_corrcoef


def _cm_panel(a, y, pred, title):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    pct = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    lab = np.array([[f"{c:,}\n({q:.1%})" for c, q in zip(r1, r2)] for r1, r2 in zip(cm, pct)])
    sns.heatmap(pct, annot=lab, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=a,
                xticklabels=["Benign", "Pathogenic"], yticklabels=["Benign", "Pathogenic"])
    a.set_title(title, fontsize=9); a.set_xlabel("predicted"); a.set_ylabel("true")


def confusion_grid(y, preds, path, dpi=200, title=""):
    """preds: name -> (calibrated_prob, threshold). One normalised confusion matrix per model."""
    names = list(preds)
    nc = 4; nr = int(np.ceil(len(names) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4.4 * nc, 4.0 * nr))
    ax = np.atleast_1d(ax).ravel()
    for a, n in zip(ax, names):
        pc, thr = preds[n]
        pred = (pc >= thr).astype(int)
        _cm_panel(a, y, pred, "\n".join(textwrap.wrap(n, 34)) + f"\nMCC={matthews_corrcoef(y, pred):.3f}  thr={thr:.2f}")
    for a in ax[len(names):]:
        a.axis("off")
    fig.suptitle(title or "Confusion matrices (pooled out-of-fold; cell = count and % of true class)", y=1.0)
    _save(fig, path, dpi)


def confusion_thresholds(y, pc, thr, strength, name, path, dpi=200):
    fpr, tpr, th = roc_curve(y, pc)
    youden = th[np.argmax(tpr - fpr)]
    spec95 = th[np.where(fpr <= 0.05)[0][-1]]
    sens90 = th[np.where(tpr >= 0.90)[0][0]]
    panels = [(f"MCC-optimal (chosen) thr={thr:.2f}", thr, np.ones(len(y), bool)),
              (f"Youden J thr={youden:.2f}", youden, np.ones(len(y), bool)),
              (f"Specificity ≥95% thr={spec95:.2f}", spec95, np.ones(len(y), bool)),
              (f"Sensitivity ≥90% thr={sens90:.2f}", sens90, np.ones(len(y), bool)),
              ("Only 'definite' labels (chosen thr)", thr, strength == "definite"),
              ("Only 'likely' labels (chosen thr)", thr, strength == "likely")]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for a, (t, th_, m) in zip(ax.ravel(), panels):
        if m.sum() == 0 or len(np.unique(y[m])) < 2:
            a.axis("off"); continue
        _cm_panel(a, y[m], (pc[m] >= th_).astype(int), t)
    fig.suptitle(f"{name}: confusion matrices at different operating points / label confidence", y=1.0)
    _save(fig, path, dpi)


def confusion_folds(res, name, path, dpi=200):
    """Mean ± SD of the normalised confusion matrix across all grouped test folds."""
    fd = res.fold_df.query("model == @name")
    M = []
    for _, r in fd.iterrows():
        rep, fo = int(r["repeat"]), int(r["fold"])
        idx = np.where(res.fold_id[rep] == fo)[0]
        p = res.oof_cal[name][rep, idx] >= r["thr"]
        yy = res.y[idx]
        if len(np.unique(yy)) < 2:
            continue
        tn = ((~p) & (yy == 0)).sum() / (yy == 0).sum(); tp = (p & (yy == 1)).sum() / (yy == 1).sum()
        M.append([[tn, 1 - tn], [1 - tp, tp]])
    M = np.array(M); mu, sd = M.mean(0), M.std(0)
    lab = np.array([[f"{mu[i, j]:.1%}\n± {sd[i, j]:.1%}" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.heatmap(mu, annot=lab, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                xticklabels=["Benign", "Pathogenic"], yticklabels=["Benign", "Pathogenic"])
    ax.set(title=f"{name}\nfold-to-fold confusion (mean ± SD, n={len(M)} folds)", xlabel="predicted", ylabel="true")
    ax.title.set_fontsize(9)
    _save(fig, path, dpi)


def metrics_heatmap(num, path, dpi=200):
    cols = [c for c in num.columns if num[c].notna().any()]
    z = (num[cols] - num[cols].min()) / (num[cols].max() - num[cols].min()).replace(0, 1)
    ann = num[cols].map(lambda v: "" if np.isnan(v) else f"{v:.3f}")
    fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 5, 0.5 * len(num) + 2))
    sns.heatmap(z.where(num[cols].notna()), annot=ann, fmt="", cmap="YlGnBu", cbar=False, ax=ax,
                linewidths=.5, linecolor="white")
    ax.set_title("Model × metric overview (colour = column-wise rank; lower is better for Brier/ECE)")
    ax.set_ylabel("")
    _save(fig, path, dpi)


def fold_stability(fold_df, order, path, dpi=200):
    fig, ax = plt.subplots(1, 3, figsize=(19, 0.45 * len(order) + 2.5), sharey=True)
    for a, m in zip(ax, ["auroc", "auprc", "mcc"]):
        d = fold_df[fold_df.model.isin(order)]
        sns.boxplot(data=d, y="model", x=m, order=order, ax=a, color="#cfe0f1", fliersize=0)
        sns.stripplot(data=d, y="model", x=m, order=order, ax=a, color="k", size=3, alpha=.6)
        a.set_title(f"{m.upper()} across all grouped test folds"); a.set_ylabel("")
    _save(fig, path, dpi)


def decision_curve(y, pcs, path, dpi=200):
    pt = np.linspace(0.05, 0.95, 91)
    prev = y.mean()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for n, p in pcs.items():
        nb = [((p >= t) & (y == 1)).mean() - ((p >= t) & (y == 0)).mean() * t / (1 - t) for t in pt]
        ax.plot(pt, nb, label=n)
    ax.plot(pt, prev - (1 - prev) * pt / (1 - pt), "k--", label="treat all")
    ax.axhline(0, color="gray", label="treat none")
    ax.set(ylim=(-0.05, prev + 0.05), xlabel="threshold probability", ylabel="net benefit",
           title="Decision-curve analysis (calibrated, out-of-fold)")
    ax.legend(fontsize=8)
    _save(fig, path, dpi)


def conformal_panel(y, pv, name, alpha, path, dpi=200):
    from .evaluate import conformal_metrics
    ok = ~np.isnan(pv).any(1)
    y, pv = y[ok], pv[ok]
    al = np.linspace(0.01, 0.5, 50)
    M = pd.DataFrame([conformal_metrics(y, pv, a) for a in al]); M["alpha"] = al
    fig, ax = plt.subplots(1, 4, figsize=(22, 5))
    ax[0].plot(1 - al, M.conf_cov, label="overall"); ax[0].plot(1 - al, M.conf_cov_path, label="pathogenic")
    ax[0].plot(1 - al, M.conf_cov_ben, label="benign"); ax[0].plot([0.5, 1], [0.5, 1], "k--")
    ax[0].set(title="A. Empirical coverage vs. target (1−α)", xlabel="target coverage 1−α", ylabel="observed coverage"); ax[0].legend()
    ax[1].stackplot(al, M.conf_singleton, M.conf_both, M.conf_empty, labels=["single label", "both (uncertain)", "empty"],
                    colors=["#59a14f", "#edc948", "#bab0ac"])
    ax[1].set(title="B. Prediction-set composition", xlabel="α", ylabel="fraction of variants"); ax[1].legend(loc="center right")
    ax[2].plot(al, M.conf_acc_singleton, color="#e45756", lw=2)
    ax[2].axhline(((pv[:, 1] > pv[:, 0]).astype(int) == y).mean(), color="k", ls="--", label="accuracy, no abstention")
    ax[2].set(title="C. Accuracy on confident (single-label) calls", xlabel="α", ylabel="accuracy"); ax[2].legend()
    S = pv > alpha
    kinds = np.where(S.sum(1) == 2, "uncertain", np.where(S.sum(1) == 0, "empty", np.where(S[:, 1], "pathogenic", "benign")))
    ct = pd.crosstab(pd.Series(np.where(y == 1, "true pathogenic", "true benign")), pd.Series(kinds), normalize="index")
    ct = ct.reindex(columns=[c for c in ["pathogenic", "benign", "uncertain", "empty"] if c in ct.columns])
    ct.plot(kind="barh", stacked=True, ax=ax[3], color=["#d95f02", "#1b9e77", "#edc948", "#bab0ac"][:ct.shape[1]])
    ax[3].set(title=f"D. Outcome at α={alpha}", xlabel="fraction"); ax[3].legend(fontsize=8)
    fig.suptitle(f"Class-conditional conformal prediction — {name}", y=1.02)
    _save(fig, path, dpi)


def subgroup_performance(df, feat, score, path, dpi=200, min_n=40):
    from .evaluate import auroc
    y = df["label"].values
    cnt = df.groupby("protein_key")["label"].transform("size")
    gr = pd.qcut(feat["grantham"], 3, labels=["conservative", "moderate", "radical"])
    groups = {
        "variants per protein": pd.cut(cnt, [0, 5, 20, 100, 1e9], labels=["≤5", "6–20", "21–100", ">100"]),
        "protein length": pd.qcut(df["seq_len"], 4, duplicates="drop"),
        "Grantham tertile": gr,
        "label confidence": df["label_strength"],
        "wild-type class": df["wt_aa"].map(lambda a: __import__("mutpred.constants", fromlist=["x"]).AA_CLASS[a]),
        "relative position": pd.cut(df["rel_pos"], [0, .1, .4, .7, .9, 1.0001]),
    }
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for a, (t, g) in zip(ax.ravel(), groups.items()):
        rows = []
        for k in pd.Series(g).dropna().unique():
            m = (pd.Series(g) == k).values
            if m.sum() >= min_n and len(np.unique(y[m])) == 2:
                rows.append((str(k), auroc(y[m], score[m]), m.sum()))
        if not rows:
            a.axis("off"); continue
        r = pd.DataFrame(rows, columns=["g", "auc", "n"]).sort_values("g")
        a.bar(r.g, r.auc, color="#4c78a8")
        for i, (v, n) in enumerate(zip(r.auc, r.n)):
            a.text(i, v + .005, f"n={n:,}", ha="center", fontsize=7)
        a.set(title=t, ylim=(max(0.4, r.auc.min() - .1), 1.0), ylabel="AUROC"); a.tick_params(axis="x", rotation=25)
    fig.suptitle("Subgroup performance (pooled out-of-fold AUROC)", y=1.0)
    _save(fig, path, dpi)


def protein_track(df, llr, proba, thr, pk, path, domains=None, dpi=200):
    sub = df[df.protein_key == pk]
    fig, ax = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    col = np.where(sub.label == 1, C_PATH, C_BEN)
    for a, v, t in [(ax[0], llr[sub.index], "ESM log-likelihood ratio (lower = more damaging)"),
                    (ax[1], proba[sub.index], "Predicted P(pathogenic)")]:
        a.vlines(sub.pos, 0 if a is ax[0] else 0, v, color=col, alpha=.6)
        a.scatter(sub.pos, v, c=col, s=28, zorder=3, edgecolor="k", lw=.3)
        a.set_ylabel(t, fontsize=8)
    ax[1].axhline(thr, color="k", ls="--", lw=.8); ax[1].set_ylim(0, 1)
    if domains is not None:
        for _, d in domains[domains.protein_id == sub.protein_id.iloc[0]].iterrows():
            for a in ax:
                a.axvspan(d.start, d.end, color="#4c78a8", alpha=.12)
            ax[0].text((d.start + d.end) / 2, ax[0].get_ylim()[1], d["name"], ha="center", va="top", fontsize=8)
    ax[1].set_xlabel("residue position")
    ax[0].set_title(f"{pk} — orange = pathogenic, green = benign (ClinVar)")
    _save(fig, path, dpi)


def shap_by_outcome(values, names, outcome, path, top=14, dpi=200):
    ok = ~np.isnan(values).any(1)
    V, o = values[ok], np.asarray(outcome)[ok]
    idx = np.argsort(np.abs(V).mean(0))[::-1][:top]
    M = pd.DataFrame({k: V[o == k][:, idx].mean(0) for k in ["TP", "TN", "FP", "FN"] if (o == k).any()},
                     index=[names[i] for i in idx])
    fig, ax = plt.subplots(figsize=(6.5, 0.4 * top + 2))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "mean SHAP"})
    ax.set_title("Mean SHAP by prediction outcome\n(why FPs/FNs go wrong)")
    _save(fig, path, dpi)


def stack_weights(wdf, path, dpi=200):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.barplot(data=wdf, y="base", x="weight", errorbar="sd", color="#4c78a8", ax=ax)
    ax.set(title="Stacking meta-learner weights (log-odds scale; mean ± SD over folds)", ylabel="")
    _save(fig, path, dpi)


def learning_curve_plot(lc, path, dpi=200):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, m in zip(ax, ["auroc", "auprc"]):
        sns.lineplot(data=lc, x="frac", y=m, hue="model", marker="o", errorbar="sd", ax=a)
        a.set(title=f"Learning curve ({m.upper()})", xlabel="fraction of training clusters used")
    _save(fig, path, dpi)


# =====================================================================  v3 additions
def _wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    m = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def risk_curves(feat, y, cols, path, nbins=10, dpi=200):
    """Pathogenic fraction (±95% Wilson CI) across quantile bins of each feature - 'does the feature carry signal?'."""
    cols = [c for c in dict.fromkeys(cols) if c in feat.columns and feat[c].notna().sum() > 100 and feat[c].nunique() > 2]
    nc = 4; nr = int(np.ceil(len(cols) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4.4 * nc, 3.4 * nr))
    ax = np.atleast_1d(ax).ravel()
    for a, c in zip(ax, cols):
        v = feat[c]
        ok = v.notna().values
        try:
            b = pd.qcut(v[ok], nbins, duplicates="drop")
        except ValueError:
            b = pd.cut(v[ok], nbins)
        g = pd.DataFrame({"b": b, "y": y[ok], "x": v[ok]}).groupby("b", observed=True).agg(x=("x", "median"), k=("y", "sum"), n=("y", "size"))
        ci = np.array([_wilson(k, n) for k, n in zip(g.k, g.n)])
        fr = g.k / g.n
        a.plot(g.x, fr, "o-", color="#4c78a8")
        a.fill_between(g.x, ci[:, 0], ci[:, 1], alpha=.25, color="#4c78a8")
        a.axhline(y.mean(), color="gray", ls="--", lw=.8)
        a.set(title=c, ylim=(0, 1), ylabel="P(pathogenic)")
    for a in ax[len(cols):]:
        a.axis("off")
    fig.suptitle("Pathogenic fraction across feature quantiles (dashed = base rate)", y=1.0)
    _save(fig, path, dpi)


def error_overlap(y, preds, path, dpi=200):
    """Jaccard overlap of the *error sets* of different models (are they wrong on the same variants?)."""
    names = list(preds)
    E = {n: set(np.where(((preds[n][0] >= preds[n][1]).astype(int)) != y)[0]) for n in names}
    M = np.array([[len(E[a] & E[b]) / max(len(E[a] | E[b]), 1) for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(0.7 * len(names) + 5, 0.6 * len(names) + 3.5))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="mako_r", xticklabels=["\n".join(textwrap.wrap(n, 18)) for n in names],
                yticklabels=names, ax=ax, vmin=0, vmax=1, cbar_kws={"label": "Jaccard of error sets"})
    ax.set_title("Error overlap between models (low = complementary → ensembling helps)")
    ax.tick_params(axis="x", labelsize=6); ax.tick_params(axis="y", labelsize=7)
    _save(fig, path, dpi)


def risk_coverage(y, pcs, path, dpi=200):
    """Selective prediction: accuracy / balanced accuracy when only the most confident x% of variants are called."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for n, (p, thr) in pcs.items():
        conf = np.abs(p - thr)
        o = np.argsort(-conf)
        pred = (p >= thr).astype(int)[o]; yy = y[o]
        cov = np.arange(1, len(y) + 1) / len(y)
        acc = np.cumsum(pred == yy) / np.arange(1, len(y) + 1)
        step = max(1, len(y) // 300)
        ax[0].plot(cov[::step], acc[::step], label=n)
        bal = []
        for c in cov[::step]:
            k = int(c * len(y)); pp, tt = pred[:k], yy[:k]
            sp = ((pp == 0) & (tt == 0)).sum() / max((tt == 0).sum(), 1); se = ((pp == 1) & (tt == 1)).sum() / max((tt == 1).sum(), 1)
            bal.append((sp + se) / 2)
        ax[1].plot(cov[::step], bal, label=n)
    ax[0].set(title="A. Accuracy vs. coverage (abstain on least-confident)", xlabel="coverage", ylabel="accuracy")
    ax[1].set(title="B. Balanced accuracy vs. coverage", xlabel="coverage", ylabel="balanced accuracy")
    ax[0].legend(fontsize=7)
    _save(fig, path, dpi)
