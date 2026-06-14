"""Energy-Based Model branch.

A conditional EBM head E_theta(z, y) over the 5 soil targets, learned on the
shared GeoSoil latent z. Trained by Noise-Contrastive Estimation (the energy of
the true target is pushed below that of noise targets). Gives:
  - predictions by gradient-minimizing energy over y (from a heteroscedastic init)
  - an OUT-OF-DISTRIBUTION score (the min energy itself)
A research-branch alternative to the heteroscedastic Gaussian head; compared on
the same frozen latent so differences are attributable to the UQ mechanism.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold

from . import config as C
from . import data as D
from .metrics import metrics


class EnergyNet(nn.Module):
    def __init__(self, z_dim: int, n_targets: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + n_targets, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def energy(self, z, y):
        return self.net(torch.cat([z, y], dim=-1)).squeeze(-1)


def train_ebm(Z, Y, mask, n_neg=16, epochs=300, lr=1e-3, device="cpu"):
    z_dim, n_t = Z.shape[1], Y.shape[1]
    model = EnergyNet(z_dim, n_t).to(device)
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    Mt = torch.as_tensor(mask, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n = len(Zt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            z, y, m = Zt[idx], Yt[idx], Mt[idx]
            b = len(idx)
            # noise targets: shuffle real y across batch + Gaussian jitter (NCE noise)
            neg = y[torch.randint(0, b, (b, n_neg), device=device)]
            neg = neg + 0.5 * torch.randn_like(neg)
            e_pos = model.energy(z, y).unsqueeze(1)                  # (b,1)
            e_neg = model.energy(z.unsqueeze(1).expand(-1, n_neg, -1).reshape(-1, z.shape[1]),
                                 neg.reshape(-1, y.shape[1])).reshape(b, n_neg)
            logits = -torch.cat([e_pos, e_neg], dim=1)              # positive should win
            loss = F.cross_entropy(logits, torch.zeros(b, dtype=torch.long, device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def predict_ebm(model, Z, y_init, steps=60, lr=0.1, device="cpu"):
    """Minimize energy over y from an init; return yhat, per-target sigma (curvature), min energy."""
    z = torch.as_tensor(Z, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_init, dtype=torch.float32, device=device).clone().requires_grad_(True)
    opt = torch.optim.Adam([y], lr=lr)
    for _ in range(steps):
        e = model.energy(z, y).sum()
        opt.zero_grad(); e.backward(); opt.step()
    yhat = y.detach()
    # local curvature -> sigma ~ 1/sqrt(d2E/dy2)
    y2 = yhat.clone().requires_grad_(True)
    e = model.energy(z, y2).sum()
    g = torch.autograd.grad(e, y2, create_graph=True)[0]
    curv = torch.zeros_like(yhat)
    for j in range(yhat.shape[1]):
        gj = torch.autograd.grad(g[:, j].sum(), y2, retain_graph=True)[0][:, j]
        curv[:, j] = gj.detach()
    sigma = 1.0 / torch.sqrt(curv.clamp(min=1e-3))
    with torch.no_grad():
        emin = model.energy(z, yhat)
    return yhat.cpu().numpy(), sigma.cpu().numpy(), emin.cpu().numpy()


def run(variant="jepa", device="cpu") -> dict:
    zp = C.RESULTS / f"oof_z_{variant}.npy"
    if not zp.exists():
        return {"error": "run training first to produce oof_z"}
    Z = np.load(zp)
    master = pd.read_parquet(C.DATA_PROC / "master.parquet")
    has = D.modality_presence(master)
    olm = master[master["source"] == "openlandmap"].reset_index(drop=True)
    norm = D.fit_normalizer(olm, has)            # global target norm for the EBM target space
    Y, M = D.transform_targets(olm, norm)
    # warm-start init: the main model's predictions, mapped into normalized target space
    pp = C.RESULTS / f"oof_pred_{variant}.npy"
    if pp.exists():
        phys = np.load(pp); init_all = np.zeros_like(Y)
        for j, t in enumerate(C.TARGETS):
            v = np.log1p(np.clip(phys[:, j], 0, None)) if t in C.LOG_TARGETS else phys[:, j]
            init_all[:, j] = (v - norm.mean[f"_tgt_{t}"]) / norm.std[f"_tgt_{t}"]
        init_all = np.nan_to_num(init_all)
    else:
        init_all = np.zeros_like(Y)
    groups = olm["block"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    oof = np.full_like(Y, np.nan); oof_sig = np.full_like(Y, np.nan)
    for tr, va in gkf.split(Z, groups=groups):
        model = train_ebm(Z[tr], Y[tr], M[tr], device=device)
        yhat, sig, _ = predict_ebm(model, Z[va], init_all[va], device=device)
        oof[va] = yhat; oof_sig[va] = sig
    res = {"variant": variant, "targets": {}, "calibration": {}}
    for j, t in enumerate(C.TARGETS):
        pred = D.invert_target(oof[:, j], t, norm)
        res["targets"][t] = metrics(olm[t].to_numpy(), pred)
        zsc = (Y[:, j] - oof[:, j]) / np.clip(oof_sig[:, j], 1e-6, None)
        res["calibration"][t] = {"cov@95%": round(float(np.mean(np.abs(zsc) <= 1.96)), 3)}
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--variant", default="jepa")
    args = ap.parse_args()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    res = run(args.variant)
    (C.RESULTS / "ebm.json").write_text(json.dumps(res, indent=2))
    if "targets" in res:
        print("== EBM head on frozen latent ==")
        for t, m in res["targets"].items():
            print(f"  {t:5s} R²={m['r2']:.3f}  cov@95%={res['calibration'][t]['cov@95%']}")
    print(res if "error" in res else "saved " + str(C.RESULTS / "ebm.json"))
