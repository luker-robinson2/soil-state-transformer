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
