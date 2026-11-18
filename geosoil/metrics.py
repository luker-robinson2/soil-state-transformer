"""Torch-free metric helpers (kept separate so tree baselines don't import torch,
which otherwise causes a duplicate-OpenMP segfault with LightGBM on macOS)."""
from __future__ import annotations

from typing import Dict

import numpy as np


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
