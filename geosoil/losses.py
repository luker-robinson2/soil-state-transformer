"""Multi-objective losses for GeoSoil.

  - supervised : masked heteroscedastic Gaussian NLL over the 5 targets
  - jepa       : cross-modal predict-in-latent (online predicts EMA target)
  - infonce    : cross-modal contrastive alignment of paired modalities
  - mask       : masked AlphaEarth feature reconstruction (denoising)
  - vicreg     : variance + covariance regularization (collapse prevention)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def gaussian_nll(mu, log_var, y, mask):
    """Masked heteroscedastic NLL; mask selects samples with a label present."""
    inv = torch.exp(-log_var)
    nll = 0.5 * (log_var + (y - mu) ** 2 * inv)
    m = mask.sum().clamp(min=1.0)
    return (nll * mask).sum() / m


def info_nce(a, b, valid, temp=0.1):
    """Symmetric InfoNCE over rows where both modalities are present."""
    idx = valid.squeeze(-1) > 0.5
    if idx.sum() < 2:
        return a.new_zeros(())
    a, b = F.normalize(a[idx], dim=-1), F.normalize(b[idx], dim=-1)
    logits = a @ b.t() / temp
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def jepa_loss(pred, target, valid):
    """Predict-in-latent: online prediction vs stop-grad EMA target (cosine MSE)."""
    idx = valid.squeeze(-1) > 0.5
    if idx.sum() < 1:
        return pred.new_zeros(())
    p = F.normalize(pred[idx], dim=-1)
    t = F.normalize(target[idx].detach(), dim=-1)
    return (2 - 2 * (p * t).sum(-1)).mean()


def vicreg(z, gamma=1.0, eps=1e-4):
    """Variance hinge + covariance off-diagonal penalty (Bardes et al., 2021)."""
    z = z - z.mean(0)
    std = torch.sqrt(z.var(0) + eps)
    var_loss = F.relu(gamma - std).mean()
    n, d = z.shape
    cov = (z.t() @ z) / (n - 1)
    cov_loss = (cov.fill_diagonal_(0) ** 2).sum() / d
    return var_loss + cov_loss


def _tok(names: List[str], name: str) -> int:
    return names.index(name)


def cross_modal_pairs(names: List[str]) -> List[Tuple[str, str]]:
    """AlphaEarth paired with every other rich modality that is present."""
    rich = [n for n in ("s2", "baresoil", "sar", "climate", "veg", "moisture", "terrain", "precip", "cdl") if n in names]
    return [("ae", r) for r in rich]


def compute_losses(model, out, batch, cfg, names: List[str]) -> Tuple[torch.Tensor, Dict[str, float]]:
    L = cfg.loss
    presence = out["presence"]
    proj = out["proj"]                                   # (B,M,proj) online
    logs: Dict[str, float] = {}

    sup = gaussian_nll(out["mu"], out["log_var"], batch["y"], batch["y_mask"])
    total = L.w_supervised * sup
    logs["sup"] = float(sup.detach())

    pairs = cross_modal_pairs(names)
    if pairs and (L.w_infonce > 0 or L.w_jepa > 0):
        tproj, _ = model.target_proj(batch)              # EMA targets (no grad)
        ince = proj.new_zeros(())
        jep = proj.new_zeros(())
        for a, b in pairs:
            ia, ib = _tok(names, a), _tok(names, b)
            valid = (presence[:, ia] * presence[:, ib]).unsqueeze(-1)
            if L.w_infonce > 0:
                ince = ince + info_nce(proj[:, ia], proj[:, ib], valid, L.infonce_temp)
            if L.w_jepa > 0:
                pred_b = model.predict_cross(proj[:, ia])
                pred_a = model.predict_cross(proj[:, ib])
                jep = jep + 0.5 * (jepa_loss(pred_b, tproj[:, ib], valid)
                                   + jepa_loss(pred_a, tproj[:, ia], valid))
        ince, jep = ince / len(pairs), jep / len(pairs)
        total = total + L.w_infonce * ince + L.w_jepa * jep
        logs["infonce"], logs["jepa"] = float(ince.detach()), float(jep.detach())

    if L.w_mask > 0:
        ae = batch["ae"]
        m = (torch.rand_like(ae) < L.mask_ratio).float()
        masked = dict(batch)
        masked["ae"] = ae * (1 - m)
        recon = model.recon_ae(masked)
        denom = m.sum().clamp(min=1.0)
        mask_loss = ((recon - ae) ** 2 * m).sum() / denom
        total = total + L.w_mask * mask_loss
        logs["mask"] = float(mask_loss.detach())

    if L.w_vicreg > 0:
        vic = vicreg(out["vic"])
        total = total + L.w_vicreg * vic
        logs["vicreg"] = float(vic.detach())

    if L.w_forecast > 0 and len(model.forecasters):
        fc = model.forecast(batch)
        total = total + L.w_forecast * fc
        logs["forecast"] = float(fc.detach())

    logs["total"] = float(total.detach())
    return total, logs
