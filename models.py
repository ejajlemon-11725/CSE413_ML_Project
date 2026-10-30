"""Model specs, estimators, in-fold tuning, calibration and threshold selection."""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:  # graceful fallback so the pipeline still runs
    xgb, HAS_XGB = None, False

G = "XGB" if HAS_XGB else "HGB"


@dataclass
class ModelSpec:
    name: str
    kind: str                       # gbm | rf | lr | zeroshot
    blocks: Tuple[str, ...]
    zero_col: Optional[str] = None  # zeroshot: column used as score
    sign: float = 1.0               # zeroshot: +1 if larger => more pathogenic
    explain: bool = False
    bases: Tuple[str, ...] = ()     # stack: names of base models


BASELINE = "RF: Bio (baseline)"
PROPOSED = f"{G}: Bio + ESM full (proposed)"
STACK = f"Stack: {G}+RF+LR (meta-LR)"


EXTRA_LABEL = {"struct": "AlphaFold", "cons": "MSA-conservation", "annot": "UniProt-annot"}


def default_specs(use_esm=True, extras=()):
    """Ablation ladder. `extras` = subset of ('struct','cons','annot') that are available."""
    extras = tuple(e for e in ("struct", "cons", "annot") if e in extras)
    s = [
        ModelSpec("Zero-shot: BLOSUM62", "zeroshot", ("bio",), "blosum62", -1),
        ModelSpec("Zero-shot: Grantham", "zeroshot", ("bio",), "grantham", +1),
        ModelSpec(BASELINE, "rf", ("bio",)),
        ModelSpec(f"{G}: Bio", "gbm", ("bio",)),
    ]
    core = ("bio",)
    if use_esm:
        core = ("bio", "esm_scalar", "esm_pca")
        s += [
            ModelSpec("Zero-shot: ESM LLR", "zeroshot", ("esm_scalar",), "esm_llr", -1),
            ModelSpec("LR: ESM LLR only", "lr", ("esm_scalar",), "esm_llr"),
            ModelSpec(f"{G}: ESM scalar", "gbm", ("esm_scalar",)),
            ModelSpec(f"{G}: ESM embeddings", "gbm", ("esm_pca",)),
            ModelSpec(f"{G}: Bio + ESM scalar", "gbm", ("bio", "esm_scalar")),
            ModelSpec(PROPOSED if not extras else f"{G}: Bio + ESM full", "gbm", core, explain=not extras),
        ]
    for e in extras:                                   # add one evidence source at a time
        s.append(ModelSpec(f"{G}: Bio{' + ESM' if use_esm else ''} + {EXTRA_LABEL[e]}", "gbm", core + (e,)))
    if extras:
        allb = core + extras
        if use_esm:
            s.append(ModelSpec(f"{G}: Bio + " + " + ".join(EXTRA_LABEL[e] for e in extras) + " (no ESM)",
                               "gbm", ("bio",) + extras))
        s.append(ModelSpec(f"{G}: ALL features (proposed)", "gbm", allb, explain=True))
        s.append(ModelSpec("RF: ALL features", "rf", allb))
    elif use_esm:
        s.append(ModelSpec("RF: Bio + ESM full", "rf", core))
    else:
        s[3].explain = True
    prop = next(x for x in s if x.explain)
    rf_full = next((x.name for x in s if x.name.startswith("RF: ") and x.name != BASELINE), BASELINE)
    bases = [prop.name, rf_full, f"{G}: Bio"] + (["LR: ESM LLR only"] if use_esm else [])
    s.append(ModelSpec(STACK, "stack", (), bases=tuple(dict.fromkeys(bases))))
    return s


def proposed_name(specs):
    return next(x.name for x in specs if x.explain)


def stack(blocks, spec):
    if spec.kind == "zeroshot":
        return blocks[spec.blocks[0]][[spec.zero_col]].values.astype(float), [spec.zero_col]
    if spec.kind == "lr" and spec.zero_col:
        return blocks[spec.blocks[0]][[spec.zero_col]].values.astype(float), [spec.zero_col]
    cols, names = [], []
    for b in spec.blocks:
        cols.append(blocks[b].values.astype(float))
        names += list(blocks[b].columns)
    return np.hstack(cols), names


def make_estimator(kind, y, seed, params=None):
    params = dict(params or {})
    pos, neg = float(np.sum(y == 1)), float(np.sum(y == 0))
    if kind == "gbm":
        if HAS_XGB:
            base = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                        colsample_bytree=0.6, min_child_weight=3, reg_lambda=2.0)
            base.update(params)
            return xgb.XGBClassifier(**base, scale_pos_weight=neg / max(pos, 1), eval_metric="logloss",
                                     tree_method="hist", random_state=seed, n_jobs=-1)
        base = dict(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=2.0, min_samples_leaf=20)
        base.update(params)
        return HistGradientBoostingClassifier(**base, class_weight="balanced", random_state=seed)
    if kind == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                                    class_weight="balanced", n_jobs=-1, random_state=seed))
    if kind == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    raise ValueError(kind)


SPACE_XGB = {"max_depth": [2, 3, 4, 5, 6], "learning_rate": [0.02, 0.03, 0.05, 0.08],
             "n_estimators": [150, 300, 500], "subsample": [0.6, 0.8, 1.0],
             "colsample_bytree": [0.4, 0.6, 0.8], "min_child_weight": [1, 3, 5, 10],
             "reg_lambda": [0.5, 1, 2, 5, 10]}
SPACE_HGB = {"max_depth": [2, 3, 4, 5, 6], "learning_rate": [0.02, 0.03, 0.05, 0.08],
             "max_iter": [150, 300, 500], "l2_regularization": [0.5, 1, 2, 5, 10],
             "min_samples_leaf": [5, 10, 20, 40]}


def _inner_splits(X, y, g, seed, k=3):
    return list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(X, y, g))


def tune(kind, X, y, g, cfg, seed):
    """Random search with an inner grouped CV (scoring: AUPRC). Runs inside each outer training fold."""
    best, best_s = None, -1
    for params in ParameterSampler(SPACE_XGB if HAS_XGB else SPACE_HGB, cfg.tune_iters, random_state=seed):
        sc = []
        for tr, va in _inner_splits(X, y, g, seed):
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                continue
            m = make_estimator(kind, y[tr], seed, params).fit(X[tr], y[tr])
            sc.append(average_precision_score(y[va], m.predict_proba(X[va])[:, 1]))
        if sc and np.mean(sc) > best_s:
            best, best_s = params, np.mean(sc)
    return best


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate_and_threshold(oof, y):
    """Platt calibration + MCC-optimal threshold from inner out-of-fold predictions."""
    ok = ~np.isnan(oof)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return None, 0.5
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(oof[ok]), y[ok])
    pc = cal.predict_proba(_logit(oof[ok]))[:, 1]
    grid = np.linspace(0.02, 0.98, 97)
    mcc = [matthews_corrcoef(y[ok], (pc >= t).astype(int)) for t in grid]
    return cal, float(grid[int(np.argmax(mcc))])


@dataclass
class Fit:
    model: object
    calibrator: object
    threshold: float
    params: Optional[dict]
    oof_raw: Optional[np.ndarray] = None   # inner out-of-fold probabilities on the training set
    oof_cal: Optional[np.ndarray] = None   # ... after Platt calibration
    coef: Optional[np.ndarray] = None      # stack: meta-learner weights


def fit_model(spec, Xtr, ytr, gtr, cfg, seed):
    params = tune(spec.kind, Xtr, ytr, gtr, cfg, seed) if (cfg.tune and spec.kind == "gbm") else None
    model = make_estimator(spec.kind, ytr, seed, params).fit(Xtr, ytr)
    cal, thr, oof, oof_cal = None, 0.5, None, None
    if cfg.threshold_method == "inner":
        oof = np.full(len(ytr), np.nan)
        for tr, va in _inner_splits(Xtr, ytr, gtr, seed):
            if len(np.unique(ytr[tr])) < 2:
                continue
            m = make_estimator(spec.kind, ytr[tr], seed, params).fit(Xtr[tr], ytr[tr])
            oof[va] = m.predict_proba(Xtr[va])[:, 1]
        cal, thr = calibrate_and_threshold(oof, ytr)
        oof_cal = oof.copy()
        if cal is not None:
            ok = ~np.isnan(oof)
            oof_cal[ok] = cal.predict_proba(_logit(oof[ok]))[:, 1]
    return Fit(model, cal, thr, params, oof, oof_cal)


def predict(fit: Fit, X):
    p = fit.model.predict_proba(X)[:, 1]
    pc = fit.calibrator.predict_proba(_logit(p))[:, 1] if fit.calibrator is not None else p
    return p, pc


def _lo(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_stack(base_fits, ytr):
    """Meta logistic regression on the *inner out-of-fold* calibrated outputs of the base models (no leakage)."""
    Z = np.column_stack([_lo(f.oof_cal if f.oof_cal is not None else f.oof_raw) for f in base_fits])
    ok = ~np.isnan(Z).any(1)
    meta = LogisticRegression(C=1.0, max_iter=1000).fit(Z[ok], ytr[ok])
    ptr = np.full(len(ytr), np.nan)
    ptr[ok] = meta.predict_proba(Z[ok])[:, 1]
    grid = np.linspace(0.02, 0.98, 97)
    thr = float(grid[int(np.argmax([matthews_corrcoef(ytr[ok], (ptr[ok] >= t).astype(int)) for t in grid]))])
    return Fit(meta, None, thr, None, ptr, ptr, meta.coef_[0])


def predict_stack(fit, base_pcs):
    p = fit.model.predict_proba(np.column_stack([_lo(p) for p in base_pcs]))[:, 1]
    return p, p
