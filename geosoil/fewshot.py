"""Few-shot local adaptation: how much do k lab samples in a place improve the global model? (foundation-model test)

A soil foundation model is a prior that a handful of local samples update. This benchmark measures that on the
saved out-of-fold predictions of the global model (baseline_c1 -> results/runs/oof_<stamp>), without retraining.

Places are cells of a given size (default 0.05°, about 5 km: farm scale; 0.25° for a district). In each held-out
cell with at least k + 5 sites, k sites are drawn as "samples" and the rest are predicted. Repeated over draws.

Estimators, all on the model's scale (log for SOC/OM/CEC/bases):
  global       the global model's prediction, unchanged (k = 0 baseline)
  local_mean   mean of the k samples (no model at all)
  offset_eb    global + shrunk mean residual of the k samples. Random-effects model: residual = cell offset u
               + noise e, u ~ (0, τ²), e ~ (0, σ²). The best linear predictor of u from k residuals is
               w * mean(residual) with w = k τ² / (k τ² + σ²) (empirical Bayes / James-Stein shrinkage). τ² and σ²
               are estimated by one-way ANOVA on the OTHER folds' cells, so nothing is tuned on the scored cell.
  idw_resid    global + inverse-distance-weighted residuals of the k samples, shrunk by the same w

Reported per target, cell size and k: MSE of each estimator relative to the global model's MSE on the same
scored sites (< 1 = better), with a cell-bootstrap 95% interval for offset_eb.

    python -m geosoil.fewshot results/runs/oof_<stamp> [--cell 0.05] [--ks 1,3,5,10,20] [--draws 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

MIN_EXTRA = 5


def variance_components(cell: np.ndarray, resid: np.ndarray) -> tuple[float, float]:
    """One-way random-effects ANOVA (method of moments): between-cell τ² and within-cell σ² of the residuals."""
    df = pd.DataFrame({"c": cell, "r": resid})
    g = df.groupby("c")["r"]
    n_i, m_i = g.size().to_numpy(float), g.mean().to_numpy()
    keep = n_i >= 2
    n_i, m_i = n_i[keep], m_i[keep]
    within = df[df["c"].isin(g.size()[g.size() >= 2].index)]
    ss_w = float(((within["r"] - within.groupby("c")["r"].transform("mean")) ** 2).sum())
    a, N = len(n_i), n_i.sum()
    if a < 3:
        return 0.0, float(np.var(resid))
    sigma2 = ss_w / (N - a)
    grand = np.sum(n_i * m_i) / N
    msb = np.sum(n_i * (m_i - grand) ** 2) / (a - 1)
    n0 = (N - np.sum(n_i ** 2) / N) / (a - 1)
    tau2 = max((msb - sigma2) / n0, 0.0)
    return float(tau2), float(sigma2)


def _idw(xy_s, r_s, xy_q, power=2.0):
    d = np.sqrt(((xy_q[:, None, :] - xy_s[None, :, :]) ** 2).sum(-1)) + 1e-6
    w = 1 / d ** power
    return (w * r_s[None, :]).sum(1) / w.sum(1)


def run(o: pd.DataFrame, cell_deg: float, ks: list[int], draws: int, seed: int = 0) -> dict:
    o = o.dropna(subset=["y", "pred", "lon", "lat"]).copy()
    o["cell"] = (np.floor(o["lon"] / cell_deg).astype(int).astype(str) + "_" +
                 np.floor(o["lat"] / cell_deg).astype(int).astype(str))
    o["resid"] = o["y"] - o["pred"]
    o = o.drop_duplicates(["site_id"])                    # one value per site (topsoil intervals collapse)
    rng = np.random.default_rng(seed)
    comps = {}
    for f in o["fold"].unique():
        other = o[o["fold"] != f]
        comps[f] = variance_components(other["cell"].to_numpy(), other["resid"].to_numpy())
    out = {"cell_deg": cell_deg, "variance_components": {str(k): v for k, v in comps.items()}}
    for k in ks:
        rows = []
        for (c, f), g in o.groupby(["cell", "fold"]):
            if len(g) < k + MIN_EXTRA:
                continue
            tau2, sigma2 = comps[f]
            w = k * tau2 / (k * tau2 + sigma2) if (k * tau2 + sigma2) > 0 else 0.0
            y, p, r = g["y"].to_numpy(), g["pred"].to_numpy(), g["resid"].to_numpy()
            xy = np.c_[g["lon"].to_numpy() * np.cos(np.radians(g["lat"].to_numpy())), g["lat"].to_numpy()]
            for _ in range(draws):
                idx = rng.permutation(len(g))
                s, q = idx[:k], idx[k:]
                rows.append({
                    "cell": c, "n": len(q),
                    "global": float(np.sum((p[q] - y[q]) ** 2)),
                    "local_mean": float(np.sum((y[s].mean() - y[q]) ** 2)),
                    "offset_eb": float(np.sum((p[q] + w * r[s].mean() - y[q]) ** 2)),
                    "idw_resid": float(np.sum((p[q] + w * _idw(xy[s], r[s], xy[q]) - y[q]) ** 2)),
                    "w": w})
        if not rows:
            continue
        d = pd.DataFrame(rows)
        per_cell = d.groupby("cell")[["global", "local_mean", "offset_eb", "idw_resid"]].sum()
        g0 = per_cell["global"].sum()
        boot = []
        cells = per_cell.to_numpy()
        for _ in range(500):
            b = cells[rng.integers(0, len(cells), len(cells))]
            boot.append(b[:, 2].sum() / b[:, 0].sum())
        out[f"k{k}"] = {"cells": int(len(per_cell)), "scored_sites_per_draw": int(d["n"].sum() / draws),
                        "mean_shrinkage_w": float(d["w"].mean()),
                        **{f"rel_mse_{e}": float(per_cell[e].sum() / g0) for e in ("local_mean", "offset_eb", "idw_resid")},
                        "rel_mse_offset_eb_lo": float(np.quantile(boot, 0.025)),
                        "rel_mse_offset_eb_hi": float(np.quantile(boot, 0.975))}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--cells", default="0.05,0.25")
    ap.add_argument("--ks", default="1,3,5,10,20")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--targets", default="")
    a = ap.parse_args(argv)
    ks = [int(x) for x in a.ks.split(",")]
    res = {}
    for t in (a.targets.split(",") if a.targets else C.TARGETS):
        f = a.run / f"{t}.parquet"
        if not f.exists():
            continue
        o = pd.read_parquet(f)
        if "lon" not in o and (C.DATA_PROC / "labels_c1.parquet").exists():
            xy = pd.read_parquet(C.DATA_PROC / "labels_c1.parquet", columns=["site_id", "lon", "lat"]).drop_duplicates("site_id")
            o = o.merge(xy, on="site_id", how="left")
        res[t] = {}
        for cd in (float(x) for x in a.cells.split(",")):
            r = run(o, cd, ks, a.draws)
            res[t][f"{cd}deg"] = r
            for k in ks:
                if f"k{k}" in r:
                    v = r[f"k{k}"]
                    print(f"{t:5s} {cd:.2f}° k={k:2d}  cells {v['cells']:5d}  w {v['mean_shrinkage_w']:.2f}  "
                          f"rel MSE: local-only {v['rel_mse_local_mean']:.2f}  offset {v['rel_mse_offset_eb']:.3f} "
                          f"[{v['rel_mse_offset_eb_lo']:.3f}, {v['rel_mse_offset_eb_hi']:.3f}]  idw {v['rel_mse_idw_resid']:.3f}",
                          flush=True)
    dest = a.run / "fewshot.json"
    dest.write_text(json.dumps(res, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
