"""Local and regional model tiers: how skill depends on how close the labels are (docs/research/LOCAL-MODELS.md).

The 1° spatial-block CV answers one question: how well does the model predict where no labels are nearby?
Regional and local models answer others, so this module measures three things on the same rows:

  scales    the global LightGBM cross-validated at 1° (~100 km), 0.25° (~25 km) and 0.05° (~5 km) blocks, each
            with and without neighbouring-label features. Smaller blocks leave labels closer to each validation
            site; the curve says how much nearby labels are worth.
  distance  out-of-fold R² binned by distance from each validation site to the nearest training site of the same
            target. This is the honest way to report local skill: by distance, not by block size.
  regional  radius models: for each anchor cell, LightGBM trained only on training rows within R km of the cell
            centre, scored on the cell's validation rows beside the global model and their average (blend).

Methods are standard digital-soil-mapping practice (spatial CV by block size: Roberts et al. 2017, Ecography
40:913; local vs global models: Lark 2000; geographically weighted models: Brunsdon et al. 1996).

    python -m geosoil.local_eval [--parts scales,distance,regional] [--max-rows 150000]
    # -> geosoil/results/runs/local_eval_<time>.json
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from .baseline_c1 import BASE, TOPSOIL, cv_scores, groups
from .corpus.splits import block_id
from .metrics import metrics

SCALES = {"1deg": ("fold", 1.0), "025deg": ("fold_025", 0.25), "005deg": ("fold_005", 0.05)}
DIST_BINS_KM = (0, 1, 5, 25, 100, 300, np.inf)
EARTH_KM = 6371.0088
RADII_KM = (150, 400)
ANCHOR_DEG = 2.0           # anchor cells for the regional models
MIN_LOCAL_ROWS = 1500      # fewer training rows within R: no regional model for that cell
MIN_CELL_ROWS = 150        # fewer validation rows in a cell: not scored


def _with_scale_block(d: pd.DataFrame, deg: float) -> pd.DataFrame:
    """Neighbour features exclude a site's own block; at a finer scale that block is the finer one."""
    d = d.copy()
    d["block"] = block_id(d["lon"], d["lat"], deg).values
    return d


def distance_to_training(d: pd.DataFrame, fold_col: str) -> np.ndarray:
    """km from each row to the nearest site of another fold (the training set when that row was validated)."""
    from sklearn.neighbors import BallTree
    out = np.full(len(d), np.nan)
    rad = np.radians(d[["lat", "lon"]].to_numpy(float))
    for k in sorted(d[fold_col].unique()):
        va = (d[fold_col] == k).to_numpy()
        tr_sites = d.loc[~va].drop_duplicates("site_id")
        if tr_sites.empty:
            continue
        tree = BallTree(np.radians(tr_sites[["lat", "lon"]].to_numpy(float)), metric="haversine")
        out[va] = tree.query(rad[va], k=1)[0][:, 0] * EARTH_KM
    return out


def by_distance(y: np.ndarray, pred: np.ndarray, dist: np.ndarray) -> dict:
    rows = {}
    for lo, hi in zip(DIST_BINS_KM[:-1], DIST_BINS_KM[1:]):
        m = (dist >= lo) & (dist < hi) & ~np.isnan(pred)
        key = f"{lo:g}-{hi:g}km" if np.isfinite(hi) else f">{lo:g}km"
        rows[key] = {**metrics(y[m], pred[m]), "rows": int(m.sum())} if m.sum() >= 50 else {"rows": int(m.sum())}
    return rows


def regional(d: pd.DataFrame, y: np.ndarray, global_pred: np.ndarray, cols: list[str], fold_col: str,
             radii=RADII_KM, seed: int = 0) -> dict:
    """Radius models per anchor cell, scored against the global out-of-fold predictions on the same rows."""
    import lightgbm as lgb
    from sklearn.neighbors import BallTree
    X = d[BASE + cols].to_numpy(np.float32)
    lat, lon = d["lat"].to_numpy(float), d["lon"].to_numpy(float)
    cell = block_id(d["lon"], d["lat"], ANCHOR_DEG).to_numpy()
    res = {}
    for R in radii:
        local = np.full(len(d), np.nan)
        n_models = 0
        for k in sorted(d[fold_col].unique()):
            va_all = (d[fold_col] == k).to_numpy()
            tr_idx = np.where(~va_all)[0]
            tree = BallTree(np.radians(np.c_[lat[tr_idx], lon[tr_idx]]), metric="haversine")
            for c in np.unique(cell[va_all]):
                va = np.where(va_all & (cell == c))[0]
                if len(va) < MIN_CELL_ROWS:
                    continue
                centre = np.radians([[lat[va].mean(), lon[va].mean()]])
                near = tr_idx[tree.query_radius(centre, r=R / EARTH_KM)[0]]
                if len(near) < MIN_LOCAL_ROWS:
                    continue
                m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.8, min_child_samples=30,
                                      random_state=seed, verbose=-1)
                m.fit(X[near], y[near])
                local[va] = m.predict(X[va])
                n_models += 1
        s = ~np.isnan(local)
        res[f"{R}km"] = {"models": n_models, "rows_scored": int(s.sum()), "share_of_rows": round(float(s.mean()), 3),
                         "global": metrics(y[s], global_pred[s]), "regional": metrics(y[s], local[s]),
                         "blend": metrics(y[s], 0.5 * (local[s] + global_pred[s]))}
    return res


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", default="scales,distance,regional")
    ap.add_argument("--targets", default=",".join(C.TARGETS))
    ap.add_argument("--max-rows", type=int, default=150_000)
    ap.add_argument("--table", default=str(C.DATA_PROC / "train_c1.parquet"))
    a = ap.parse_args(argv)
    parts = set(a.parts.split(","))
    df = pd.read_parquet(a.table)
    df = df[df["interval"].isin(TOPSOIL)].copy()
    df["depth_mid"] = (df["depth_top"] + df["depth_bot"]) / 2
    runs = C.RESULTS / "runs"
    runs.mkdir(exist_ok=True)
    out = runs / f"local_eval_{pd.Timestamp.now():%Y%m%dT%H%M}.json"
    res = {"radii_km": list(RADII_KM), "distance_bins_km": [str(b) for b in DIST_BINS_KM]}
    for t in a.targets.split(","):
        dt = df[df["target"] == t]
        if a.max_rows and len(dt) > a.max_rows:
            dt = dt.sample(a.max_rows, random_state=0)
        cols = groups(dt.columns)["all"]
        res[t] = {}
        for name, (fold_col, deg) in SCALES.items():
            if fold_col not in dt or (not {"scales", "distance"} & parts and name != "025deg"):
                continue
            ds = _with_scale_block(dt, deg)
            sc, d, y, pred = cv_scores(ds, t, cols, fold_col=fold_col, return_pred=True)
            entry = {"all": sc}
            if "scales" in parts:
                entry["all+neighbours"] = cv_scores(ds, t, cols, neighbours=True, fold_col=fold_col)
            if "distance" in parts:
                dist = distance_to_training(d, fold_col)
                entry["median_km_to_training"] = round(float(np.nanmedian(dist)), 2)
                entry["by_distance"] = by_distance(y, pred, dist)
            if "regional" in parts and name == "025deg":
                entry["regional"] = regional(d, y, pred, cols, fold_col)
            res[t][name] = entry
            print(f"{t:5s} {name:7s} R2 {sc['r2']:.3f}"
                  + (f"  +nb {entry['all+neighbours']['r2']:.3f}" if "all+neighbours" in entry else "")
                  + (f"  median km to training {entry.get('median_km_to_training')}" if "distance" in parts else ""),
                  flush=True)
            out.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
