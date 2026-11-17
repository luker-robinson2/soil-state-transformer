"""Neighbouring-label features, built inside each cross-validation fold (roadmap item I1).

Soil properties are strongly spatially autocorrelated, so the measured values at nearby sites are among
the most informative predictors in digital soil mapping (Hengl et al. 2018, PeerJ 6:e5518, "RFsp";
Sekulić et al. 2020, Remote Sensing 12:1687). They are also the easiest features to leak, so the rules are:

  - Neighbours come only from the training rows of the current fold, and only for the same target and
    depth interval.
  - A validation site's neighbours all lie in other spatial blocks (folds are whole 1° blocks). Training
    rows therefore skip neighbours in their own block, so the model learns from the same neighbour
    distances it will see at test time. Without this, training rows would have neighbours metres
    away and the model would over-trust them.
  - At deployment every label is available and neighbours are closer than in CV, a shift in the
    conservative direction.

Columns (values in the target's model space, e.g. log1p for SOC):
  nb_idw      inverse-distance-weighted mean of the k nearest values
  nb_near     value of the nearest neighbour
  nb_std      spread of the k values
  nb_d1_km    distance to the nearest neighbour
  nb_dmean_km mean distance to the k neighbours
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_KM = 6371.0088
K = 8
COLUMNS = ("nb_idw", "nb_near", "nb_std", "nb_d1_km", "nb_dmean_km")


def _rad(df: pd.DataFrame) -> np.ndarray:
    return np.radians(df[["lat", "lon"]].to_numpy(float))


def _knn_excluding_block(tree, train_block: np.ndarray, q: np.ndarray, q_block, k: int):
    """k nearest training points whose block differs from each query's block (None = no exclusion)."""
    n = len(train_block)
    dist = np.full((len(q), k), np.nan)
    idx = np.full((len(q), k), -1, dtype=np.int64)
    todo = np.arange(len(q))
    want = k if q_block is None else min(n, 4 * k)
    while len(todo):
        d, i = tree.query(q[todo], k=min(n, want))
        if q_block is not None:
            ok = train_block[i] != np.asarray(q_block)[todo][:, None]
        else:
            ok = np.ones_like(i, dtype=bool)
        enough = ok.sum(axis=1) >= min(k, n)
        last = want >= n
        for r in np.where(enough | last)[0]:
            sel = np.where(ok[r])[0][:k]
            dist[todo[r], :len(sel)] = d[r, sel]
            idx[todo[r], :len(sel)] = i[r, sel]
        todo = todo[~(enough | last)]
        want = min(n, want * 8)
    return dist * EARTH_KM, idx


def summarise(dist_km: np.ndarray, vals: np.ndarray) -> pd.DataFrame:
    """Per-row features from the k neighbour distances (km) and values (NaN where fewer than k exist)."""
    w = 1.0 / np.maximum(dist_km, 0.05)                  # 50 m floor: co-located sites don't dominate
    w = np.where(np.isnan(vals), 0.0, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        idw = np.nansum(w * np.nan_to_num(vals), axis=1) / w.sum(axis=1)
    return pd.DataFrame({
        "nb_idw": np.where(w.sum(axis=1) > 0, idw, np.nan),
        "nb_near": vals[:, 0],
        "nb_std": np.nanstd(vals, axis=1) if vals.shape[1] > 1 else np.nan,
        "nb_d1_km": dist_km[:, 0],
        "nb_dmean_km": np.nanmean(dist_km, axis=1),
    })


def fold_features(train: pd.DataFrame, query: pd.DataFrame, y_train: np.ndarray,
                  exclude_same_block: bool, k: int = K) -> pd.DataFrame:
    """Neighbour features for `query` rows from `train` rows of the same depth interval.

    train/query need lon, lat, interval and block columns; y_train is aligned with train.
    exclude_same_block=True for the training rows themselves, False for validation/test rows.
    """
    from sklearn.neighbors import BallTree
    out = pd.DataFrame(np.nan, index=query.index, columns=list(COLUMNS))
    for iv, qi in query.groupby("interval").groups.items():
        tmask = (train["interval"] == iv).to_numpy() & ~np.isnan(y_train)
        if tmask.sum() == 0:
            continue
        tr = train[tmask]
        yv = y_train[tmask]
        tree = BallTree(_rad(tr), metric="haversine")
        q = query.loc[qi]
        qb = q["block"].to_numpy() if exclude_same_block else None
        d, i = _knn_excluding_block(tree, tr["block"].to_numpy(), _rad(q), qb, k)
        vals = np.where(i >= 0, yv[np.clip(i, 0, None)], np.nan)
        out.loc[qi] = summarise(d, vals).to_numpy()
    return out
