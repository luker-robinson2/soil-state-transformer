"""Edaphic zones (model tier T2, docs/research/LOCAL-MODELS.md): do models per pedo-climatic zone beat one model?

Zones are clusters of sites in covariate space, the published pedometric approach (McBratney & de Gruijter 1992,
J. Soil Sci. 43:159; Hengl et al. 2004 for continuous classes): robust-scale the features, project to principal
components, and k-means the sites. Zone assignment uses covariates only, never labels, so it carries no label
leakage; models are still scored on the fixed 1° spatial folds.

For each k, on the same rows:
  global          the LightGBM baseline (all groups)
  zone_feature    the same model with the zone as an extra categorical feature
  per_zone        one model per zone, trained on that zone's training rows (falls back to global for small zones)
  blend           average of global and per_zone
  leave_zone_out  train on all other zones, test on one: how well the model transfers to an unseen pedo-climate,
                  as R² pooled over held-out zones (compare with global's R² on the same rows)

    python -m geosoil.zones [--k 8,16,32] [--targets soc,ph,sand,clay,bd]
    # -> geosoil/results/runs/zones_<time>.json
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
from .metrics import metrics

N_PCS = 16
MIN_ZONE_ROWS = 2000


def assign_zones(sites: pd.DataFrame, cols: list[str], k: int, seed: int = 0) -> np.ndarray:
    """Zone id per site from robust-scaled covariates -> PCA -> k-means (missing values filled with the median)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    X = sites[cols].to_numpy(float)
    med = np.nanmedian(X, axis=0)
    q = np.nanpercentile(X, [25, 75], axis=0)
    iqr = np.where((q[1] - q[0]) > 0, q[1] - q[0], 1.0)
    Z = np.nan_to_num((X - med) / iqr, nan=0.0).clip(-10, 10)
    P = PCA(n_components=min(N_PCS, Z.shape[1]), random_state=seed).fit_transform(Z)
    return MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=4096).fit_predict(P)


def _lgb(seed=0):
    import lightgbm as lgb
    return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=seed, verbose=-1)


LZO_MAX_K = 8   # leave-zone-out costs one fit per (zone, fold); run it for the coarsest zoning only


def zone_scores(d: pd.DataFrame, y: np.ndarray, global_pred: np.ndarray, cols: list[str], zone: np.ndarray,
                seed: int = 0, leave_zone_out: bool = True) -> dict:
    X = d[BASE + cols].to_numpy(np.float32)
    folds = d["fold"].to_numpy()
    zf = np.full(len(d), np.nan)
    per = np.full(len(d), np.nan)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        Xz = np.c_[X, zone].astype(np.float32)
        m = _lgb(seed).fit(Xz[tr], y[tr], categorical_feature=[Xz.shape[1] - 1])
        zf[va] = m.predict(Xz[va])
        for z in np.unique(zone[va]):
            trz, vaz = tr & (zone == z), va & (zone == z)
            if trz.sum() < MIN_ZONE_ROWS:
                per[vaz] = global_pred[vaz]
                continue
            per[vaz] = _lgb(seed).fit(X[trz], y[trz]).predict(X[vaz])
    # leave one zone out: every row predicted by a model that never saw its zone (nor its 1° fold, to stay spatial)
    lzo = np.full(len(d), np.nan)
    for z in (np.unique(zone) if leave_zone_out else []):
        for k in np.unique(folds):
            te = (zone == z) & (folds == k)
            if not te.any():
                continue
            trn = (zone != z) & (folds != k)
            lzo[te] = _lgb(seed).fit(X[trn], y[trn]).predict(X[te])
    sizes = pd.Series(zone).value_counts()
    return {"global": metrics(y, global_pred), "zone_feature": metrics(y, zf), "per_zone": metrics(y, per),
            "blend": metrics(y, 0.5 * (global_pred + per)),
            **({"leave_zone_out": metrics(y, lzo), "global_same_rows": metrics(y, global_pred)} if leave_zone_out else {}),
            "zone_rows_min_median_max": [int(sizes.min()), int(sizes.median()), int(sizes.max())]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", default="8,16,32")
    ap.add_argument("--targets", default="soc,ph,sand,clay,bd")
    ap.add_argument("--max-rows", type=int, default=150_000)
    ap.add_argument("--table", default=str(C.DATA_PROC / "train_c1.parquet"))
    a = ap.parse_args(argv)
    df = pd.read_parquet(a.table)
    df = df[df["interval"].isin(TOPSOIL)].copy()
    df["depth_mid"] = (df["depth_top"] + df["depth_bot"]) / 2
    cols_all = groups(df.columns)["all"]
    # zones from site covariates (one row per site), excluding depth and coordinates
    sites = df.drop_duplicates("site_id")[["site_id"] + cols_all]
    zones = {k: pd.Series(assign_zones(sites, cols_all, k), index=sites["site_id"].to_numpy()) for k in map(int, a.k.split(","))}
    runs = C.RESULTS / "runs"
    runs.mkdir(exist_ok=True)
    out = runs / f"zones_{pd.Timestamp.now():%Y%m%dT%H%M}.json"
    res = {"k": list(zones), "n_pcs": N_PCS, "min_zone_rows": MIN_ZONE_ROWS}
    for t in a.targets.split(","):
        dt = df[df["target"] == t]
        if a.max_rows and len(dt) > a.max_rows:
            dt = dt.sample(a.max_rows, random_state=0)
        _, d, y, pred = cv_scores(dt, t, cols_all, return_pred=True)
        res[t] = {}
        for k, zs in zones.items():
            z = zs.reindex(d["site_id"]).to_numpy()
            r = zone_scores(d, y, pred, cols_all, z, leave_zone_out=k <= LZO_MAX_K)
            res[t][f"k{k}"] = r
            print(f"{t:5s} k={k:<3d} global {r['global']['r2']:.3f}  zone-feature {r['zone_feature']['r2']:.3f}  "
                  f"per-zone {r['per_zone']['r2']:.3f}  blend {r['blend']['r2']:.3f}  "
                  + (f"leave-zone-out {r['leave_zone_out']['r2']:.3f}" if "leave_zone_out" in r else ""), flush=True)
            out.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
