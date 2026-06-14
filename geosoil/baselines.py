"""Strong tabular baselines under the SAME spatial-block CV — the bar to beat.

Ridge, RandomForest, LightGBM, XGBoost, CatBoost, TabPFN-v2, plus the
"neural-encoder -> GBM" hybrid (frozen GeoSoil latent z as features into LightGBM).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # avoid duplicate-OpenMP segfault on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config as C
from .metrics import metrics

warnings.filterwarnings("ignore")


def feature_matrix(df: pd.DataFrame, has: dict) -> np.ndarray:
    cols: List[str] = list(C.AE_COLS)
    if has.get("s2"):
        cols += list(C.S2_COLS)
    if has.get("terrain"):
        cols += list(C.TERRAIN_COLS)
    if has.get("baresoil"):
        cols += list(C.BARESOIL_COLS)
    if has.get("sar"):
        cols += list(C.SAR_COLS)
    if has.get("precip"):
        cols += list(C.PRECIP_COLS)
    if has.get("cdl"):
        cols += list(C.CDL_COLS)
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    # Trees can't ingest sequences — give them per-variable summary stats (mean/min/max)
    extra = []
    for present, vars_ in (("climate", C.CLIMATE_VARS), ("veg", C.VEG_VARS), ("moisture", C.MOISTURE_VARS)):
        if has.get(present):
            for v in vars_:
                seq = df[[f"{v}_{i:02d}" for i in range(C.SEQ_LEN)]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
                extra += [np.nanmean(seq, 1, keepdims=True), np.nanmin(seq, 1, keepdims=True), np.nanmax(seq, 1, keepdims=True)]
    if extra:
        X = np.hstack([X] + extra)
    return X


def _impute(Xtr, Xva):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xva), med, Xva)


def get_models():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    m = {
        "ridge": lambda: Ridge(alpha=10.0),
        "rf": lambda: RandomForestRegressor(n_estimators=400, n_jobs=1, random_state=0),
        "lightgbm": lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                                              min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                                              reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=1, num_threads=1),
        "xgboost": lambda: xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5,
                                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                            random_state=0, n_jobs=1, nthread=1),
        "catboost": lambda: CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6,
                                              l2_leaf_reg=3.0, random_seed=0, verbose=0, thread_count=1),
    }
    return m


def run(quick: bool = False, with_tabpfn: bool = True) -> dict:
    master = pd.read_parquet(C.DATA_PROC / "master.parquet")
    has = D_presence(master)
    olm = master[master["source"] == "openlandmap"].reset_index(drop=True)
    X = feature_matrix(olm, has)
    groups = olm["block"].to_numpy()
    gkf = GroupKFold(n_splits=C.TrainConfig.n_spatial_folds)
    splits = list(gkf.split(olm, groups=groups))

    models = get_models()
    if quick:
        models = {k: models[k] for k in ("ridge", "lightgbm")}
    results: Dict[str, dict] = {}

    def log_target(t, v):
        return np.log1p(np.clip(v, 0, None)) if t in C.LOG_TARGETS else v

    def inv_target(t, v):
        return np.expm1(v) if t in C.LOG_TARGETS else v

    for name, make in models.items():
        results[name] = {}
        for t in C.TARGETS:
            y = pd.to_numeric(olm[t], errors="coerce").to_numpy(np.float64)
            oof = np.full(len(olm), np.nan)
            for tr, va in splits:
                ytr = log_target(t, y[tr])
                ok = np.isfinite(ytr)
                Xtr, Xva = _impute(X[tr], X[va])
                mdl = make()
                mdl.fit(Xtr[ok], ytr[ok])
                oof[va] = inv_target(t, mdl.predict(Xva))
            results[name][t] = metrics(y, oof)
        print(f"[{name:9s}] " + " ".join(f"{t}={results[name][t]['r2']:.3f}" for t in C.TARGETS))

    if with_tabpfn and not quick:
        try:
            from tabpfn import TabPFNRegressor
            results["tabpfn"] = {}
            for t in C.TARGETS:
                y = pd.to_numeric(olm[t], errors="coerce").to_numpy(np.float64)
                oof = np.full(len(olm), np.nan)
                for tr, va in splits:
                    ytr = log_target(t, y[tr]); ok = np.isfinite(ytr)
                    Xtr, Xva = _impute(X[tr], X[va])
                    reg = TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
                    reg.fit(Xtr[ok], ytr[ok])
                    oof[va] = inv_target(t, reg.predict(Xva))
                results["tabpfn"][t] = metrics(y, oof)
            print("[tabpfn   ] " + " ".join(f"{t}={results['tabpfn'][t]['r2']:.3f}" for t in C.TARGETS))
        except Exception as e:
            print("tabpfn skipped:", str(e)[:200])

    return results


def D_presence(master):
    from .data import modality_presence
    return modality_presence(master)


def hybrid_run(variant: str = "jepa") -> dict:
    """Hybrid: feed the FROZEN GeoSoil latent z into LightGBM (and z + raw features).
    Tests whether the learned representation carries signal beyond raw modalities
    that a strong GBM can exploit."""
    import lightgbm as lgb
    from .data import modality_presence
    zp = C.RESULTS / f"oof_z_{variant}.npy"
    if not zp.exists():
        return {"error": "run training first (need oof_z)"}
    Z = np.load(zp)
    master = pd.read_parquet(C.DATA_PROC / "master.parquet")
    has = modality_presence(master)
    olm = master[master["source"] == "openlandmap"].reset_index(drop=True)
    Xraw = feature_matrix(olm, has)
    Zf = np.hstack([Z, np.nan_to_num(Xraw)])
    groups = olm["block"].to_numpy()
    gkf = GroupKFold(n_splits=C.TrainConfig.n_spatial_folds)
    splits = list(gkf.split(olm, groups=groups))
    mk = lambda: lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31,
                                   min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                                   reg_lambda=1.0, random_state=0, verbose=-1, n_jobs=1, num_threads=1)
    out = {"z_only": {}, "z_plus_features": {}}
    for feats, key in ((Z, "z_only"), (Zf, "z_plus_features")):
        for t in C.TARGETS:
            y = pd.to_numeric(olm[t], errors="coerce").to_numpy(np.float64)
            yl = np.log1p(np.clip(y, 0, None)) if t in C.LOG_TARGETS else y
            oof = np.full(len(olm), np.nan)
            for tr, va in splits:
                m = mk(); m.fit(feats[tr], yl[tr])
                p = m.predict(feats[va])
                oof[va] = np.expm1(p) if t in C.LOG_TARGETS else p
            out[key][t] = metrics(y, oof)
        print(f"[hybrid {key:16s}] " + " ".join(f"{t}={out[key][t]['r2']:.3f}" for t in C.TARGETS))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-tabpfn", action="store_true")
    ap.add_argument("--hybrid", action="store_true", help="run frozen-latent->LightGBM hybrid")
    args = ap.parse_args()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    if args.hybrid:
        res = hybrid_run()
        (C.RESULTS / "hybrid.json").write_text(json.dumps(res, indent=2))
        print("saved", C.RESULTS / "hybrid.json")
    else:
        res = run(quick=args.quick, with_tabpfn=not args.no_tabpfn)
        (C.RESULTS / "baselines.json").write_text(json.dumps(res, indent=2))
        print("saved", C.RESULTS / "baselines.json")
