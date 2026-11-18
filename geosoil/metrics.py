"""Torch-free metric helpers (kept separate so tree baselines don't import torch,
which otherwise causes a duplicate-OpenMP segfault with LightGBM on macOS)."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    if len(yt) < 3:
        return {"r2": float("nan"), "rmse": float("nan"), "rpiq": float("nan"), "n": int(len(yt))}
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2)) or 1e-9
    rmse = float(np.sqrt(ss_res / len(yt)))
    iqr = float(np.subtract(*np.percentile(yt, [75, 25])))
    return {"r2": 1 - ss_res / ss_tot, "rmse": rmse,
            "rpiq": (iqr / rmse if rmse > 0 else float("nan")), "n": int(len(yt))}


# Public class bands, so a score says whether a prediction lands in the right class, not only how close it is.
# pH: USDA Soil Survey Manual reaction classes. The others are coarse, commonly used bands (SOC g/kg, sand and
# clay %, bulk density g/cm3) chosen to separate soils that are managed differently.
CLASS_BANDS = {
    "ph": (4.5, 5.05, 5.55, 6.05, 6.55, 7.35, 7.85, 8.45, 9.05),
    "soc": (5.0, 10.0, 20.0, 40.0, 80.0),
    "sand": (15.0, 50.0, 85.0),
    "clay": (10.0, 20.0, 35.0, 60.0),
    "bd": (1.0, 1.2, 1.4, 1.6),
}


def class_agreement(target: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Share of predictions in the lab value's class, and within one class of it (values in physical units)."""
    bands = CLASS_BANDS.get(target)
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(yt) & np.isfinite(yp)
    if bands is None or ok.sum() < 3:
        return {"class_exact": float("nan"), "class_within_one": float("nan")}
    ct, cp = np.digitize(yt[ok], bands), np.digitize(yp[ok], bands)
    return {"class_exact": float((ct == cp).mean()), "class_within_one": float((np.abs(ct - cp) <= 1).mean())}


def conformal_coverage(y_true: np.ndarray, y_pred: np.ndarray, folds: np.ndarray, level: float = 0.9) -> Dict[str, float]:
    """Split-conformal intervals on spatial folds (Vovk et al. 2005; CV+ style).

    For each fold, the interval half-width is the `level` quantile of the absolute out-of-fold residuals of the
    other folds, so it is set without seeing that fold. Returns the measured coverage on held-out rows (target:
    `level`) and the mean half-width, in the units the model is fitted in.
    """
    yt, yp, f = np.asarray(y_true, float), np.asarray(y_pred, float), np.asarray(folds)
    ok = np.isfinite(yt) & np.isfinite(yp)
    res = np.abs(yt - yp)
    inside, widths = [], []
    for k in np.unique(f[ok]):
        cal, val = ok & (f != k), ok & (f == k)
        if cal.sum() < 20 or val.sum() == 0:
            continue
        n = cal.sum()
        q = np.quantile(res[cal], min(1.0, np.ceil((n + 1) * level) / n))
        inside.append(res[val] <= q)
        widths.append(np.full(val.sum(), q))
    if not inside:
        return {"coverage": float("nan"), "half_width": float("nan"), "level": level}
    return {"coverage": float(np.concatenate(inside).mean()), "half_width": float(np.concatenate(widths).mean()),
            "level": level}


# ---------------------------------------------------------------------------------------------------------------------
# Evaluation suite (docs/research/METRICS.md). R² = 1 - SSE/SST ranks models correctly only on identical rows: on a
# fixed test set SST is constant, so R² and MSE order models the same. Across row sets it mixes model change with
# test-set variance change, so comparisons use paired squared-error differences with spatial-block uncertainty,
# and skill is reported in physical units and against the lab noise floor.

def point_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Point accuracy with the bias / scale / correlation split of the MSE (Theil decomposition).

    me: mean error (bias); mae; rmse; r2 (model efficiency); r2_pearson (squared correlation: blind to bias and
    scale, never use it alone); ccc (Lin's concordance: precision x accuracy); slope (calibration slope of y on p:
    1 = right spread, <1 = predictions too spread out); mse_bias/mse_scale/mse_corr sum to mse.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return {"n": int(len(y))}
    e = p - y
    my, mp, sy, sp = y.mean(), p.mean(), y.std(), p.std()
    r = float(np.corrcoef(y, p)[0, 1]) if sy > 0 and sp > 0 else float("nan")
    mse = float(np.mean(e ** 2))
    ccc = 2 * r * sy * sp / (sy ** 2 + sp ** 2 + (my - mp) ** 2) if (sy > 0 or sp > 0) else float("nan")
    slope = float(np.cov(y, p)[0, 1] / p.var()) if p.var() > 0 else float("nan")
    return {"n": int(len(y)), "me": float(e.mean()), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt(mse)),
            "r2": 1 - mse / float(y.var()) if y.var() > 0 else float("nan"), "r2_pearson": r ** 2, "ccc": float(ccc),
            "slope": slope, "mse": mse, "mse_bias": float((mp - my) ** 2), "mse_scale": float((sp - r * sy) ** 2),
            "mse_corr": float((1 - r ** 2) * sy ** 2)}


def floor_skill(y: np.ndarray, p: np.ndarray, nugget: float) -> Dict[str, float]:
    """Share of the achievable reduction in error, on the SAME test rows: (s² - MSE) / (s² - τ²).

    s² is the variance of these test rows and τ² the lab + micro-scale noise (the noise-floor nugget). 1 = as good as
    repeat lab measurements allow; 0 = no better than the mean. excess_rmse = sqrt(MSE - τ²) is the error left to
    remove, in physical units.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    s2, mse = float(np.var(y[ok])), float(np.mean((p[ok] - y[ok]) ** 2))
    return {"floor_skill": (s2 - mse) / (s2 - nugget) if s2 > nugget else float("nan"),
            "excess_rmse": float(np.sqrt(max(mse - nugget, 0.0))), "noise_rmse": float(np.sqrt(nugget))}


def block_bootstrap(blocks: np.ndarray, stat, *arrays, n_boot: int = 1000, seed: int = 0, level: float = 0.95):
    """CI for stat(*arrays) resampling whole spatial blocks (errors are spatially correlated, so rows are not iid)."""
    rng = np.random.default_rng(seed)
    b = np.asarray(blocks)
    ub, inv = np.unique(b, return_inverse=True)
    idx_by = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    est = float(stat(*arrays))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = np.concatenate([idx_by[j] for j in rng.integers(0, len(ub), len(ub))])
        draws[i] = stat(*[a[pick] for a in arrays])
    lo, hi = np.nanquantile(draws, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"estimate": est, "lo": float(lo), "hi": float(hi), "se": float(np.nanstd(draws)), "blocks": int(len(ub))}


def paired_compare(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, blocks: np.ndarray, n_boot: int = 2000,
                   seed: int = 0) -> Dict[str, float]:
    """Is model B better than model A on the same rows? Difference in MSE (A - B; >0 means B is better).

    Uncertainty from whole spatial blocks: a block-bootstrap CI and a sign-flip randomization test on per-block mean
    loss differences (exchangeable under H0 that the models are equally good), which respects spatial correlation.
    """
    y, a, b = (np.asarray(v, float) for v in (y, p_a, p_b))
    ok = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    d = (a[ok] - y[ok]) ** 2 - (b[ok] - y[ok]) ** 2
    blk = np.asarray(blocks)[ok]
    ci = block_bootstrap(blk, np.mean, d, n_boot=n_boot, seed=seed)
    per = pd.Series(d).groupby(blk).agg(["sum", "size"])
    rng = np.random.default_rng(seed + 1)
    obs = per["sum"].sum() / per["size"].sum()
    signs = rng.choice([-1.0, 1.0], size=(n_boot, len(per)))
    null = (signs * per["sum"].to_numpy()).sum(axis=1) / per["size"].sum()
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_boot + 1))
    base = float(np.mean((a[ok] - y[ok]) ** 2))
    return {"mse_a": base, "mse_b": float(np.mean((b[ok] - y[ok]) ** 2)), "delta_mse": float(obs),
            "delta_lo": ci["lo"], "delta_hi": ci["hi"], "rel_gain": float(obs / base) if base > 0 else float("nan"),
            "p_signflip": p, "blocks": ci["blocks"], "n": int(ok.sum())}


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.1) -> float:
    """Interval (Winkler) score, a proper scoring rule for central (1 - alpha) intervals (Gneiting & Raftery 2007):
    width plus 2/alpha times the miss distance. Lower is better; rewards narrow intervals only if they still cover."""
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    return float(np.mean((hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)))


def weighted_kappa(target: str, y: np.ndarray, p: np.ndarray) -> float:
    """Quadratic-weighted Cohen's kappa on the public class bands: agreement beyond chance, penalising far misses more
    than adjacent ones (physical units)."""
    bands = CLASS_BANDS.get(target)
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    if bands is None or ok.sum() < 3:
        return float("nan")
    a, b = np.digitize(y[ok], bands), np.digitize(p[ok], bands)
    k = len(bands) + 1
    obs = np.zeros((k, k))
    np.add.at(obs, (a, b), 1)
    obs /= obs.sum()
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    den = (w * exp).sum()
    return float(1 - (w * obs).sum() / den) if den > 0 else float("nan")


def smearing_factor(y_log: np.ndarray, p_log: np.ndarray) -> float:
    """Duan (1983) smearing estimate: expm1(p) is the conditional median of a log1p model, biased low for the mean;
    mean ≈ (1 + expm1(p)) * mean(exp(residuals)) - 1."""
    r = np.asarray(y_log, float) - np.asarray(p_log, float)
    return float(np.nanmean(np.exp(r)))
