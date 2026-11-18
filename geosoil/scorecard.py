"""Scorecard: the statistical evaluation suite on saved out-of-fold predictions (docs/research/METRICS.md).

baseline_c1.py writes out-of-fold predictions for the all-groups model to results/runs/oof_<stamp>/<target>.parquet.
This module computes, per target, without retraining:
  point     ME (bias), MAE, RMSE, R² (model efficiency), squared Pearson r, Lin's CCC, calibration slope, and the
            bias/scale/correlation split of the MSE, on the model scale and (log targets) on the original scale
            using the conditional median (expm1) and the Duan-smeared mean
  floor     share of the achievable error reduction on these rows, (s² - MSE)/(s² - τ²), and the excess RMSE above
            the lab noise floor (noise_floor.json nugget)
  cis       block-bootstrap 95% CIs (resampling 1° blocks) for RMSE and R²
  interval  split-conformal 90% intervals per fold: coverage and interval score
  classes   exact / within-one class agreement and quadratic-weighted kappa (public bands)
  by        RMSE, ME and R² by source (and any other OOF column), with n

and paired comparisons between two runs on their shared rows (difference in MSE, block-bootstrap CI, sign-flip test).

    python -m geosoil.scorecard results/runs/oof_<stamp>                       # one run
    python -m geosoil.scorecard results/runs/oof_<A> --vs results/runs/oof_<B>  # paired: is B better than A?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import metrics as M


def _nugget(target: str) -> float | None:
    f = C.RESULTS / "noise_floor.json"
    if not f.exists():
        return None
    nf = json.loads(f.read_text())["targets"]
    key = f"{'soc' if target == 'om' else target}/topsoil_0_30/{'log1p' if target in C.LOG_TARGETS else 'raw'}"
    v = nf.get(key, {}).get("nugget")
    if v is not None and target == "om":
        v = float(v)                           # OM = SOC x 1.724: same nugget on the log scale (additive shift)
    return v


def conformal_intervals(y: np.ndarray, p: np.ndarray, folds: np.ndarray, level: float = 0.9):
    """Per-row split-conformal bounds: half-width from the other folds' absolute residuals."""
    lo, hi = np.full(len(y), np.nan), np.full(len(y), np.nan)
    res = np.abs(y - p)
    for k in np.unique(folds):
        cal, val = folds != k, folds == k
        n = cal.sum()
        if n < 20:
            continue
        q = np.quantile(res[cal], min(1.0, np.ceil((n + 1) * level) / n))
        lo[val], hi[val] = p[val] - q, p[val] + q
    return lo, hi


def score(o: pd.DataFrame, n_boot: int = 500) -> dict:
    t = o["target"].iloc[0]
    y, p = o["y"].to_numpy(float), o["pred"].to_numpy(float)
    blocks, folds = o["block"].astype(str).to_numpy(), o["fold"].to_numpy()
    out = {"n": int(len(o)), "blocks": int(pd.Series(blocks).nunique()), "point": M.point_metrics(y, p)}
    nug = _nugget(t)
    if nug is not None:
        out["floor"] = M.floor_skill(y, p, nug)
    out["cis"] = {
        "rmse": M.block_bootstrap(blocks, lambda a, b: np.sqrt(np.mean((a - b) ** 2)), y, p, n_boot=n_boot),
        "r2": M.block_bootstrap(blocks, lambda a, b: 1 - np.mean((a - b) ** 2) / np.var(a), y, p, n_boot=n_boot),
    }
    lo, hi = conformal_intervals(y, p, folds)
    ok = np.isfinite(lo)
    out["interval"] = {"coverage": float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))),
                       "interval_score": M.interval_score(y[ok], lo[ok], hi[ok]), "mean_width": float(np.mean(hi[ok] - lo[ok]))}
    if t in C.LOG_TARGETS:
        yo, med = np.expm1(y), np.expm1(p)
        mean = (1 + med) * M.smearing_factor(y, p) - 1
        out["original_scale"] = {"median": M.point_metrics(yo, med), "smeared_mean": M.point_metrics(yo, mean)}
        yc, pc = yo, med
    else:
        yc, pc = y, p
    out["classes"] = {**M.class_agreement(t, yc, pc), "weighted_kappa": M.weighted_kappa(t, yc, pc)}
    by = {}
    for col in ("source", "method_family"):
        if col in o:
            g = o.assign(e2=(p - y) ** 2, e=p - y).groupby(col)
            by[col] = {k: {"n": int(len(v)), "rmse": float(np.sqrt(v["e2"].mean())), "me": float(v["e"].mean()),
                           "r2": float(1 - v["e2"].mean() / v["y"].var()) if len(v) > 2 and v["y"].var() > 0 else None}
                       for k, v in g if len(v) >= 50}
    out["by"] = by
    return out


def compare(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Paired comparison on the rows both runs predicted (same site, interval and fold)."""
    keys = ["site_id", "interval"]
    m = a.merge(b[keys + ["pred", "y"]], on=keys, suffixes=("_a", "_b"))
    m = m[np.isclose(m["y_a"], m["y_b"])]
    return M.paired_compare(m["y_a"].to_numpy(float), m["pred_a"].to_numpy(float), m["pred_b"].to_numpy(float),
                            m["block"].astype(str).to_numpy(), n_boot=n_boot)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="results/runs/oof_<stamp> directory")
    ap.add_argument("--vs", type=Path, help="a second oof directory: paired comparison, is it better than `run`?")
    ap.add_argument("--targets", default="")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args(argv)
    targets = [t for t in (a.targets.split(",") if a.targets else C.TARGETS) if (a.run / f"{t}.parquet").exists()]
    res = {}
    for t in targets:
        o = pd.read_parquet(a.run / f"{t}.parquet")
        res[t] = score(o, a.n_boot)
        if a.vs and (a.vs / f"{t}.parquet").exists():
            res[t]["vs"] = compare(o, pd.read_parquet(a.vs / f"{t}.parquet"), max(a.n_boot, 1000))
        pm, ci = res[t]["point"], res[t]["cis"]
        print(f"{t:5s} RMSE {pm['rmse']:.3f} [{ci['rmse']['lo']:.3f}, {ci['rmse']['hi']:.3f}]  R2 {pm['r2']:.3f} "
              f"[{ci['r2']['lo']:.3f}, {ci['r2']['hi']:.3f}]  CCC {pm['ccc']:.3f}  ME {pm['me']:+.3f}  "
              + (f"floor {res[t]['floor']['floor_skill']:.2f}  " if "floor" in res[t] else "")
              + (f"| vs dMSE {res[t]['vs']['delta_mse']:+.4f} [{res[t]['vs']['delta_lo']:+.4f}, "
                 f"{res[t]['vs']['delta_hi']:+.4f}] p={res[t]['vs']['p_signflip']:.3f}" if "vs" in res[t] else ""),
              flush=True)
    out = a.run / ("scorecard" + (f"_vs_{a.vs.name}" if a.vs else "") + ".json")
    out.write_text(json.dumps(res, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
