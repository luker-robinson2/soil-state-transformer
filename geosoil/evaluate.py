"""Verification suite — the crux of the goal: real, independently-checkable signal.

  - KSSL lab-truth validation : train on OpenLandMap, test on lab-measured KSSL
  - frozen-latent probes       : linear + kNN probe of z (representation quality)
  - cross-modal retrieval      : recall@k AlphaEarth<->Sentinel-2 (did contrast work?)
  - calibration                : coverage of the predictive intervals + conformal
  - figures                    : pred-vs-obs, UMAP of z, calibration curve
"""
from __future__ import annotations

import argparse
import json
from typing import Dict

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from .metrics import metrics


# --------------------------------------------------------------------------- #
def kssl_validation(cfg: C.GeoSoilConfig, variant="jepa", epochs=None, device=None) -> dict:
    """Train on OpenLandMap; validate on lab-measured KSSL test split."""
    import numpy as np
    from .train import train_full, predict
    model, norm, has, device = train_full(cfg, variant, epochs=epochs, device=device)
    master = pd.read_parquet(C.DATA_PROC / "master.parquet")
    kssl = master[(master["source"] == "kssl") & (master["split"] == "test")].reset_index(drop=True)
    if len(kssl) == 0:
        kssl = master[master["source"] == "kssl"].reset_index(drop=True)
    tensors = D.make_tensors(kssl, norm, has)
    mu, lv, z = predict(model, tensors, np.arange(len(kssl)), device)
    res = {"n": int(len(kssl)), "targets": {}}
    for j, t in enumerate(C.TARGETS):
        pred = D.invert_target(mu[:, j], t, norm)
        res["targets"][t] = metrics(pd.to_numeric(kssl[t], errors="coerce").to_numpy(), pred)
    return res


def kssl_internal(variant="jepa") -> dict:
    """Control: in-domain lab predictability. Train on KSSL *train* split, test on
    KSSL *test* split using AlphaEarth features (Ridge + LightGBM). Separates the
    OpenLandMap->KSSL domain gap from genuine lab-signal availability."""
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from sklearn.linear_model import Ridge
    import lightgbm as lgb
    master = pd.read_parquet(C.DATA_PROC / "master.parquet")
    k = master[master["source"] == "kssl"]
    tr = k[k["split"].isin(["train", "val"])]; te = k[k["split"] == "test"]
    Xtr = tr[list(C.AE_COLS)].to_numpy(np.float32); Xte = te[list(C.AE_COLS)].to_numpy(np.float32)
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "ridge": {}, "lightgbm": {}}
    for t in C.TARGETS:
        ytr = pd.to_numeric(tr[t], errors="coerce").to_numpy(np.float64)
        yte = pd.to_numeric(te[t], errors="coerce").to_numpy(np.float64)
        if t in C.LOG_TARGETS:
            ytr_f = np.log1p(np.clip(ytr, 0, None))
        else:
            ytr_f = ytr
        ok = np.isfinite(ytr_f)
        for name, mk in (("ridge", lambda: Ridge(alpha=10.0)),
                         ("lightgbm", lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03,
                          num_leaves=31, min_child_samples=30, reg_lambda=1.0, verbose=-1,
                          n_jobs=1, num_threads=1))):
            m = mk(); m.fit(Xtr[ok], ytr_f[ok])
            pred = m.predict(Xte)
            if t in C.LOG_TARGETS:
                pred = np.expm1(pred)
            out[name][t] = metrics(yte, pred)
    return out


def probe_frozen_latent(variant="jepa") -> dict:
    """Linear (Ridge) + kNN probe of the frozen out-of-fold latent z."""
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import GroupKFold
    zp = C.RESULTS / f"oof_z_{variant}.npy"
    if not zp.exists():
        return {"error": "no oof_z; run training first"}
    Z = np.load(zp)
    master = pd.read_parquet(C.DATA_PROC / "master.parquet")
    olm = master[master["source"] == "openlandmap"].reset_index(drop=True)
    groups = olm["block"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    out = {"linear": {}, "knn": {}}
    for t in C.TARGETS:
        y = pd.to_numeric(olm[t], errors="coerce").to_numpy(np.float64)
        yl = np.log1p(np.clip(y, 0, None)) if t in C.LOG_TARGETS else y
        for name, mk in (("linear", lambda: Ridge(alpha=10.0)),
                         ("knn", lambda: KNeighborsRegressor(n_neighbors=10, weights="distance"))):
            oof = np.full(len(olm), np.nan)
            for tr, va in gkf.split(Z, groups=groups):
                m = mk(); m.fit(Z[tr], yl[tr]); oof[va] = m.predict(Z[va])
            pred = np.expm1(oof) if t in C.LOG_TARGETS else oof
            out[name][t] = metrics(y, pred)
    return out


def cross_modal_retrieval(cfg: C.GeoSoilConfig, variant="jepa", epochs=None, device=None) -> dict:
    """Recall@k that AlphaEarth retrieves its true Sentinel-2 partner (and vice versa)."""
    import torch
    from .train import train_full, DictDataset, move
    model, norm, has, device = train_full(cfg, variant, epochs=epochs, device=device)
    if "s2" not in model.names:
        return {"error": "no S2 modality"}
    master = pd.read_parquet(C.DATA_PROC / "master.parquet")
    olm = master[master["source"] == "openlandmap"].reset_index(drop=True)
    tensors = D.make_tensors(olm, norm, has)
    paired = np.where(tensors["s2_present"][:, 0] > 0.5)[0]
    paired = paired[: min(1000, len(paired))]
    ds = DictDataset(tensors, paired)
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
    ia, is2 = model.names.index("ae"), model.names.index("s2")
    A, S = [], []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            out = model(move(batch, device))
            p = torch.nn.functional.normalize(out["proj"], dim=-1)
            A.append(p[:, ia].cpu()); S.append(p[:, is2].cpu())
    A = torch.cat(A); S = torch.cat(S)
    sim = A @ S.t()
    n = sim.size(0)
    truth = torch.arange(n)
    res = {"n": int(n)}
    for k in (1, 5, 10):
        topk = sim.topk(k, dim=1).indices
        r_ab = (topk == truth[:, None]).any(1).float().mean().item()
        topk2 = sim.t().topk(k, dim=1).indices
        r_ba = (topk2 == truth[:, None]).any(1).float().mean().item()
        res[f"recall@{k}"] = round(0.5 * (r_ab + r_ba), 4)
    res["random_baseline@10"] = round(10.0 / n, 4)
    return res


def calibration(variant="jepa") -> dict:
    """Coverage of the predictive intervals + split-conformal coverage, from the
    fold-consistent standardized residuals (z-scores)."""
    zp = C.RESULTS / f"oof_zscore_{variant}.npy"
    if not zp.exists():
        return {"error": "no oof_zscore; run training first"}
    Z = np.load(zp)
    out = {}
    for j, t in enumerate(C.TARGETS):
        z = Z[:, j]; z = z[np.isfinite(z)]
        if len(z) < 10:
            continue
        cov68 = float(np.mean(np.abs(z) <= 1.0))
        cov95 = float(np.mean(np.abs(z) <= 1.96))
        half = len(z) // 2
        q = np.quantile(np.abs(z[:half]), 0.9)            # split-conformal calibration
        conf_cov = float(np.mean(np.abs(z[half:]) <= q))  # target 0.90
        out[t] = {"cov@68%": round(cov68, 3), "cov@95%": round(cov95, 3),
                  "conformal_cov@90%": round(conf_cov, 3)}
    return out


def make_figures(variant="jepa"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mp = C.RESULTS / f"oof_pred_{variant}.npy"
    if not mp.exists():
        return
    pred_all = np.load(mp); Z = np.load(C.RESULTS / f"oof_z_{variant}.npy")
    master = pd.read_parquet(C.DATA_PROC / "master.parquet")
    olm = master[master["source"] == "openlandmap"].reset_index(drop=True)

    # pred vs obs
    fig, axes = plt.subplots(1, len(C.TARGETS), figsize=(4 * len(C.TARGETS), 3.6))
    for j, t in enumerate(C.TARGETS):
        pred = pred_all[:, j]
        obs = pd.to_numeric(olm[t], errors="coerce").to_numpy()
        ax = axes[j]
        ax.scatter(obs, pred, s=4, alpha=0.3)
        lo, hi = np.nanpercentile(obs, 1), np.nanpercentile(obs, 99)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        m = metrics(obs, pred)
        ax.set_title(f"{t}  R²={m['r2']:.2f}")
        ax.set_xlabel("observed"); ax.set_ylabel("predicted")
    fig.tight_layout(); fig.savefig(C.RESULTS / f"pred_obs_{variant}.png", dpi=110); plt.close(fig)

    # UMAP of latent colored by SOC
    try:
        import umap
        emb = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=0).fit_transform(Z)
        soc = np.log1p(pd.to_numeric(olm["soc"], errors="coerce").to_numpy())
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=soc, s=5, cmap="viridis")
        plt.colorbar(sc, label="log(1+SOC)")
        ax.set_title("GeoSoil latent (UMAP) colored by SOC")
        fig.tight_layout(); fig.savefig(C.RESULTS / f"umap_latent_{variant}.png", dpi=110); plt.close(fig)
    except Exception as e:
        print("umap skipped:", str(e)[:150])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="jepa")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = C.GeoSoilConfig()
    if args.quick:
        cfg.train.ensemble_seeds = (0,)
        args.epochs = args.epochs or 8
    C.RESULTS.mkdir(parents=True, exist_ok=True)

    report: Dict[str, dict] = {}
    print("== KSSL lab-truth validation (OpenLandMap-trained -> lab, domain transfer) ==")
    report["kssl_transfer"] = kssl_validation(cfg, args.variant, args.epochs, args.device)
    for t, m in report["kssl_transfer"]["targets"].items():
        print(f"  {t:5s} R²={m['r2']:.3f} RMSE={m['rmse']:.3f} n={m['n']}")
    print("== KSSL in-domain control (train on KSSL lab -> test KSSL lab) ==")
    report["kssl_internal"] = kssl_internal(args.variant)
    for t in C.TARGETS:
        print(f"  {t:5s} ridge R²={report['kssl_internal']['ridge'][t]['r2']:.3f}  "
              f"lgbm R²={report['kssl_internal']['lightgbm'][t]['r2']:.3f}")
    print("== frozen-latent probe =="); report["probe"] = probe_frozen_latent(args.variant)
    if "linear" in report["probe"]:
        for t in C.TARGETS:
            print(f"  {t:5s} linear R²={report['probe']['linear'][t]['r2']:.3f}  kNN R²={report['probe']['knn'][t]['r2']:.3f}")
    print("== cross-modal retrieval =="); report["retrieval"] = cross_modal_retrieval(cfg, args.variant, args.epochs, args.device)
    print(" ", report["retrieval"])
    print("== calibration =="); report["calibration"] = calibration(args.variant)
    print(" ", report["calibration"])
    make_figures(args.variant)
    (C.RESULTS / f"verification_{args.variant}.json").write_text(json.dumps(report, indent=2))
    print("saved", C.RESULTS / f"verification_{args.variant}.json")


if __name__ == "__main__":
    main()
