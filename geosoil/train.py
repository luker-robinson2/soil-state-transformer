"""Training + spatial-block cross-validation for the GeoSoil bake-off.

Produces REAL, leakage-free metrics:
  - 5-fold spatial-block CV on the OpenLandMap-labelled set (multi-target)
  - out-of-fold predictions -> R2 / RMSE / RPIQ per target in physical units
  - frozen latent z exported for probing / retrieval / visualization
"""
from __future__ import annotations

import argparse
import gc
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from . import config as C
from . import data as D
from .losses import compute_losses
from .metrics import metrics
from .model import GeoSoilModel


# --------------------------------------------------------------------------- #
class DictDataset(torch.utils.data.Dataset):
    def __init__(self, tensors: dict, idx: np.ndarray):
        self.t = {k: torch.as_tensor(v[idx]) for k, v in tensors.items()}
        self.n = len(idx)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.t.items()}


def move(batch: dict, device: str) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


def train_one(tensors: dict, train_idx: np.ndarray, cfg: C.GeoSoilConfig, has: dict,
              device: str, seed: int = 0, epochs: int | None = None, verbose: bool = False) -> GeoSoilModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GeoSoilModel(cfg, has).to(device)
    if cfg.train.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    cuda = device == "cuda"
    dl = torch.utils.data.DataLoader(
        DictDataset(tensors, train_idx), batch_size=min(cfg.train.batch_size, len(train_idx)),
        shuffle=True, drop_last=len(train_idx) > cfg.train.batch_size,
        num_workers=cfg.train.num_workers if cuda else 0, pin_memory=cuda)
    epochs = epochs or cfg.train.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(dl)))
    # mixed precision (bf16 needs no scaler; fp16 does) — cuda only
    prec = cfg.train.precision if cuda else "fp32"
    amp_dtype = torch.bfloat16 if prec == "bf16" else torch.float16
    use_amp = prec in ("bf16", "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=(prec == "fp16"))
    model.train()
    for ep in range(epochs):
        last = {}
        for batch in dl:
            batch = move(batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
                loss, logs = compute_losses(model, out, batch, cfg, model.names)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            model.ema_update(cfg.loss.ema_decay)
            last = logs
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"  ep{ep:3d} " + " ".join(f"{k}={v:.3f}" for k, v in last.items()))
    return model


@torch.no_grad()
def predict(model: GeoSoilModel, tensors: dict, idx: np.ndarray, device: str):
    model.eval()
    ds = DictDataset(tensors, idx)
    dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=False)
    mus, lvs, zs = [], [], []
    for batch in dl:
        out = model(move(batch, device))
        mus.append(out["mu"].cpu().numpy())
        lvs.append(out["log_var"].cpu().numpy())
        zs.append(out["z"].cpu().numpy())
    return np.concatenate(mus), np.concatenate(lvs), np.concatenate(zs)


def train_full(cfg: C.GeoSoilConfig, variant: str = "jepa", epochs: int | None = None,
               device: str | None = None, seed: int = 0):
    """Train one model on ALL OpenLandMap rows (for KSSL/RaCA external validation)."""
    if variant == "mamba":
        cfg.model.temporal_encoder = "mamba"
    device = device or C.resolve_device(cfg.train.device)
    master = pd.read_parquet(C.DATA_PROC / "master.parquet")
    has = D.modality_presence(master)
    olm = master[master["source"] == "openlandmap"].reset_index(drop=True)
    norm = D.fit_normalizer(olm, has)
    tensors = D.make_tensors(olm, norm, has)
    model = train_one(tensors, np.arange(len(olm)), cfg, has, device, seed=seed, epochs=epochs)
    return model, norm, has, device


def spatial_cv(cfg: C.GeoSoilConfig, variant: str = "jepa", epochs: int | None = None,
               device: str | None = None, verbose: bool = True,
               source: str = "openlandmap", tag: str | None = None) -> dict:
    cfg.model.temporal_encoder = "mamba" if variant == "mamba" else cfg.model.temporal_encoder
    device = device or C.resolve_device(cfg.train.device)
    tag = tag or variant
    master = pd.read_parquet(C.DATA_PROC / "master.parquet")
    olm = master[master["source"] == source].reset_index(drop=True)
    has = D.modality_presence(olm)            # modalities available for THIS source

    gkf = GroupKFold(n_splits=cfg.train.n_spatial_folds)
    groups = olm["block"].to_numpy()
    n = len(olm)
    oof_pred = np.full((n, len(C.TARGETS)), np.nan, dtype=np.float32)   # PHYSICAL units (fold-inverted)
    oof_zscore = np.full((n, len(C.TARGETS)), np.nan, dtype=np.float32)  # standardized residual (fold-consistent)
    oof_z = np.zeros((n, cfg.model.latent_dim), dtype=np.float32)

    for fold, (tr, va) in enumerate(gkf.split(olm, groups=groups)):
        norm = D.fit_normalizer(olm.iloc[tr], has)            # feature + target norm from THIS fold's train only
        tensors = D.make_tensors(olm, norm, has)
        yv, yv_mask = D.transform_targets(olm.iloc[va], norm)  # val targets in this fold's norm space
        mus, vars_ = [], []
        for s in cfg.train.ensemble_seeds:
            model = train_one(tensors, tr, cfg, has, device, seed=s, epochs=epochs,
                              verbose=verbose and s == cfg.train.ensemble_seeds[0])
            mu, lv, z = predict(model, tensors, va, device)
            mus.append(mu); vars_.append(np.exp(lv))
            if s == cfg.train.ensemble_seeds[0]:
                oof_z[va] = z
            del model                                  # free memory between ensemble members
            gc.collect()
            if device == "mps":
                torch.mps.empty_cache()
        mus = np.stack(mus); vars_ = np.stack(vars_)
        mu_ens = mus.mean(0)
        sigma = np.sqrt(vars_.mean(0) + mus.var(0))           # aleatoric + epistemic
        # invert to physical units with THIS fold's normalizer (correct for regional folds)
        for j, t in enumerate(C.TARGETS):
            oof_pred[va, j] = D.invert_target(mu_ens[:, j], t, norm)
            oof_zscore[va, j] = np.where(yv_mask[:, j] > 0.5, (yv[:, j] - mu_ens[:, j]) / np.clip(sigma[:, j], 1e-6, None), np.nan)
        if verbose:
            mt = metrics(olm.iloc[va]["soc"].to_numpy(), oof_pred[va, 0])
            print(f"[fold {fold}] SOC r2={mt['r2']:.3f} rmse={mt['rmse']:.2f} n={mt['n']}")

    results = {"variant": variant, "source": source, "device": device,
               "modalities": {k: bool(v) for k, v in has.items()}, "n": int(n), "targets": {}}
    for j, t in enumerate(C.TARGETS):
        results["targets"][t] = metrics(olm[t].to_numpy(), oof_pred[:, j])
    np.save(C.RESULTS / f"oof_z_{tag}.npy", oof_z)
    np.save(C.RESULTS / f"oof_pred_{tag}.npy", oof_pred)
    np.save(C.RESULTS / f"oof_zscore_{tag}.npy", oof_zscore)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="jepa", choices=["jepa", "mamba", "ebm", "transformer"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--quick", action="store_true", help="few epochs / 1 seed smoke test")
    ap.add_argument("--source", default="openlandmap", help="label source: openlandmap | kssl")
    ap.add_argument("--size", default="base", choices=list(C.MODEL_PRESETS), help="model size preset")
    ap.add_argument("--precision", default=None, help="fp32 | bf16 | fp16 (cuda AMP)")
    args = ap.parse_args()
    cfg = C.GeoSoilConfig.preset(args.size)
    if args.precision:
        cfg.train.precision = args.precision
    if args.variant == "transformer":
        cfg.model.temporal_encoder = "transformer"
    if args.quick:
        cfg.train.ensemble_seeds = (0,)
        args.epochs = args.epochs or 8
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.size == "base" else f"_{args.size}"
    tag = (args.variant if args.source == "openlandmap" else f"{args.variant}_{args.source}") + sfx
    res = spatial_cv(cfg, args.variant, epochs=args.epochs, device=args.device, source=args.source, tag=tag)
    out = C.RESULTS / f"cv_{tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print("\n=== Spatial-block CV (out-of-fold) ===")
    for t, m in res["targets"].items():
        print(f"  {t:5s}  R2={m['r2']:.3f}  RMSE={m['rmse']:.3f}  RPIQ={m['rpiq']:.2f}  n={m['n']}")
    print("saved", out)


if __name__ == "__main__":
    main()
