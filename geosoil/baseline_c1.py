"""First C1 baseline: LightGBM on lab labels with the fixed spatial folds (Phase 2).

For each target, topsoil intervals (0-5, 5-15, 15-30 cm) are modelled together, with interval
depth as a feature. Feature groups are compared one at a time and together (each adds lon/lat and depth):
    every group in geosoil/features/groups.py alone | terrain+alphaearth | all
Groups whose columns are absent are skipped. Every group is scored on the SAME rows (rows with
at least one AlphaEarth or terrain value), so the columns are directly comparable.
Scores come from 5-fold spatial-block CV on `split == "train"` and are reported beside the
noise-floor R2 ceiling. The `test` blocks are not touched here; they are kept for model selection.
SOC is modelled as log1p, as everywhere else in GeoSoil.

    python -m geosoil.baseline_c1 [--max-rows 200000] [--neighbours]   # -> geosoil/results/runs/baseline_c1_<time>.json
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
from .metrics import class_agreement, conformal_coverage, metrics

TOPSOIL = ("0-5", "5-15", "15-30")
# Standard-method families per target (harmonize.py: converted pH 8.2 CEC and Mehlich-3 K count as canonical).
# --canonical-only restricts labels to these, to test whether mixed extraction methods are hurting a target.
CANONICAL = {"cec": {"cec_ph7", "cec_sum_cations_ph82"}, "k": {"nh4oac_ph7", "mehlich3"},
             "ca": {"nh4oac_ph7"}, "mg": {"nh4oac_ph7"}}
OOF_COLS = ["site_id", "block", "fold", "interval", "source", "method_family"]
BASE = ["depth_mid", "lon", "lat"]


def load_topsoil(path=None, train_only: bool = False) -> pd.DataFrame:
    """Topsoil rows of the training table, read with a parquet filter and float32 features.

    Reading the whole table and filtering afterwards needed >48 GB once the corpus passed 300k sites with landscape
    context (2026-11-18 OOM); the filter and float32 keep it to a fraction of that.
    """
    filters = [("interval", "in", list(TOPSOIL))] + ([("split", "==", "train")] if train_only else [])
    df = pd.read_parquet(path or C.DATA_PROC / "train_c1.parquet", filters=filters)
    keep64 = {"lon", "lat", "value", "depth_top", "depth_bot"}
    f64 = [c for c in df.columns if df[c].dtype == "float64" and c not in keep64]
    df[f64] = df[f64].astype("float32")
    return df.reset_index(drop=True)


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


def cv_scores(df: pd.DataFrame, target: str, cols: list[str], seed: int = 0, neighbours: bool = False,
              fold_col: str = "fold", return_pred: bool = False):
    """Spatial-CV score. neighbours=True adds fold-built neighbouring-label features (features/neighbours.py).

    fold_col picks the block scale (fold = 1°, fold_025, fold_005; see corpus/splits.py). With return_pred the
    out-of-fold predictions and the scored rows come back too (local_eval.py bins them by distance).
    """
    import lightgbm as lgb
    from .features import neighbours as NB
    d = df[(df["target"] == target) & (df["split"] == "train")].dropna(subset=["value"]).reset_index(drop=True)
    y = np.log1p(d["value"].to_numpy()) if target in C.LOG_TARGETS else d["value"].to_numpy()
    X = d[BASE + cols].to_numpy(np.float32)
    pred = np.full(len(d), np.nan)
    for k in sorted(d[fold_col].unique()):
        tr, va = (d[fold_col] != k).to_numpy(), (d[fold_col] == k).to_numpy()
        Xtr, Xva = X[tr], X[va]
        if neighbours:
            ytr = y[tr]
            nb_tr = NB.fold_features(d[tr], d[tr], ytr, exclude_same_block=True).to_numpy(np.float32)
            nb_va = NB.fold_features(d[tr], d[va], ytr, exclude_same_block=False).to_numpy(np.float32)
            Xtr, Xva = np.hstack([Xtr, nb_tr]), np.hstack([Xva, nb_va])
        m = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        m.fit(Xtr, y[tr])
        pred[va] = m.predict(Xva)
    out = metrics(y, pred)
    out["n"], out["sites"] = int(len(d)), int(d["site_id"].nunique())
    # uncertainty and class skill (metrics.py): interval coverage in model space, classes in physical units
    out.update(conformal_coverage(y, pred, d[fold_col].to_numpy()))
    phys = (lambda v: np.expm1(v)) if target in C.LOG_TARGETS else (lambda v: v)
    out.update(class_agreement(target, phys(y), phys(pred)))
    return (out, d, y, pred) if return_pred else out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-rows", type=int, default=0, help="subsample rows per target (speed)")
    ap.add_argument("--neighbours", action="store_true", help="also score fold-built neighbouring-label features")
    a = ap.parse_args(argv)
    df = load_topsoil()
    df["depth_mid"] = (df["depth_top"] + df["depth_bot"]) / 2
    floor = json.loads((C.RESULTS / "noise_floor.json").read_text())["targets"]
    res = {}
    # Runs land in results/runs/ (gitignored) so cluster jobs never dirty the git checkout;
    # copy a run to results/baseline_c1.json deliberately when it becomes the reference.
    runs = C.RESULTS / "runs"
    runs.mkdir(exist_ok=True)
    out = runs / f"baseline_c1_{pd.Timestamp.now():%Y%m%dT%H%M}.json"
    canon = set(filter(None, a.canonical_only.split(",")))
    oof_dir = runs / f"oof_{out.stem.split('_', 2)[-1]}"
    for t in (a.targets.split(",") if a.targets else C.TARGETS):
        dt = df[df["target"] == t]
        if t in canon and "method_family" in dt:
            dt = dt[dt["method_family"].isin(CANONICAL[t])]
        if a.max_rows and len(dt) > a.max_rows:
            dt = dt.sample(a.max_rows, random_state=0)
        key = f"{t}/topsoil_0_30/{'log1p' if t in C.LOG_TARGETS else 'raw'}"
        res[t] = {"r2_ceiling": floor.get(key, {}).get("r2_ceiling")}
        core = [c for c in list(C.AE_COLS) + list(terrain.FEATURES) if c in dt]
        dt = dt[dt[core].notna().any(axis=1)]                       # same rows for every group
        gs = groups(dt.columns)
        runs_ = [(g, cols, False) for g, cols in gs.items()]
        if a.neighbours:                                             # roadmap I1: alone and on top of all
            runs_ += [("neighbours", [], True), ("all+neighbours", gs["all"], True)]
        if a.groups:
            runs_ = [r for r in runs_ if r[0] in set(a.groups.split(","))]
        for g, cols, nb in runs_:
            if g == "all":                                # keep out-of-fold predictions: geosoil/scorecard.py
                res[t][g], d, y, pred = cv_scores(dt, t, cols, neighbours=nb, return_pred=True)
                oof_dir.mkdir(exist_ok=True)
                o = d[[c for c in OOF_COLS if c in d]].copy()
                o["target"], o["y"], o["pred"] = t, y, pred
                o.to_parquet(oof_dir / f"{t}.parquet", index=False)
                continue
            res[t][g] = cv_scores(dt, t, cols, neighbours=nb)
            r = res[t][g]
            print(f"{t:5s} {g:20s} R2 {r['r2']:.3f}  (ceiling {res[t]['r2_ceiling']})  n {r['n']}  "
                  f"90% cover {r['coverage']:.2f}  class {r['class_exact']:.2f}/{r['class_within_one']:.2f}", flush=True)
        out.write_text(json.dumps(res, indent=2))   # after every target: a walltime kill keeps what finished
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
