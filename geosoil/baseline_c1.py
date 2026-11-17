"""First C1 baseline: LightGBM on lab labels with the fixed spatial folds (Phase 2).

For each target, topsoil intervals (0-5, 5-15, 15-30 cm) are modelled together, with interval
depth as a feature. Feature groups are compared one at a time and together (each adds lon/lat and depth):
    every group in geosoil/features/groups.py alone | terrain+alphaearth | all
Groups whose columns are absent are skipped. Every group is scored on the SAME rows (rows with
at least one AlphaEarth or terrain value), so the columns are directly comparable.
Scores come from 5-fold spatial-block CV on `split == "train"` and are reported beside the
noise-floor R2 ceiling. The `test` blocks are not touched here; they are kept for model selection.
SOC is modelled as log1p, as everywhere else in GeoSoil.

    python -m geosoil.baseline_c1 [--max-rows 200000]    # -> geosoil/results/runs/baseline_c1_<time>.json
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import json

import numpy as np
import pandas as pd

from . import config as C
from .features import terrain
from .metrics import metrics

TOPSOIL = ("0-5", "5-15", "15-30")
BASE = ["depth_mid", "lon", "lat"]


def groups(columns) -> dict[str, list[str]]:
    """Each modality alone, terrain + AlphaEarth, and everything (location and depth are always added)."""
    from .features.groups import feature_groups
    g = feature_groups(columns, include_location=False)
    clim = g.pop("climate_annual", []) + g.pop("climate_monthly", [])
    if clim:
        g["climate"] = clim
    out = dict(g)
    if g.get("terrain") or g.get("alphaearth"):
        out["terrain+alphaearth"] = g.get("terrain", []) + g.get("alphaearth", [])
    out["all"] = [c for v in g.values() for c in v]
    return {k: v for k, v in out.items() if v}


def cv_scores(df: pd.DataFrame, target: str, cols: list[str], seed: int = 0) -> dict:
    import lightgbm as lgb
    d = df[(df["target"] == target) & (df["split"] == "train")].dropna(subset=["value"])
    y = np.log1p(d["value"].to_numpy()) if target in C.LOG_TARGETS else d["value"].to_numpy()
    X = d[BASE + cols].to_numpy(np.float32)
    pred = np.full(len(d), np.nan)
    for k in sorted(d["fold"].unique()):
        tr, va = (d["fold"] != k).to_numpy(), (d["fold"] == k).to_numpy()
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        m.fit(X[tr], y[tr])
        pred[va] = m.predict(X[va])
    out = metrics(y, pred)
    out["n"], out["sites"] = int(len(d)), int(d["site_id"].nunique())
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-rows", type=int, default=0, help="subsample rows per target (speed)")
    a = ap.parse_args(argv)
    df = pd.read_parquet(C.DATA_PROC / "train_c1.parquet")
    df = df[df["interval"].isin(TOPSOIL)].copy()
    df["depth_mid"] = (df["depth_top"] + df["depth_bot"]) / 2
    floor = json.loads((C.RESULTS / "noise_floor.json").read_text())["targets"]
    res = {}
    for t in C.TARGETS:
        dt = df[df["target"] == t]
        if a.max_rows and len(dt) > a.max_rows:
            dt = dt.sample(a.max_rows, random_state=0)
        key = f"{t}/topsoil_0_30/{'log1p' if t in C.LOG_TARGETS else 'raw'}"
        res[t] = {"r2_ceiling": floor.get(key, {}).get("r2_ceiling")}
        core = [c for c in list(C.AE_COLS) + list(terrain.FEATURES) if c in dt]
        dt = dt[dt[core].notna().any(axis=1)]                       # same rows for every group
        for g, cols in groups(dt.columns).items():
            res[t][g] = cv_scores(dt, t, cols)
            print(f"{t:5s} {g:20s} R2 {res[t][g]['r2']:.3f}  (ceiling {res[t]['r2_ceiling']})  n {res[t][g]['n']}")
    # Runs land in results/runs/ (gitignored) so cluster jobs never dirty the git checkout;
    # copy a run to results/baseline_c1.json deliberately when it becomes the reference.
    runs = C.RESULTS / "runs"
    runs.mkdir(exist_ok=True)
    out = runs / f"baseline_c1_{pd.Timestamp.now():%Y%m%dT%H%M}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
