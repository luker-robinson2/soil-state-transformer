"""Temporal-forecasting objective (dynamics awareness).

To make the latent predictive of *field alterations* and *micro-events* rather
than only static soil state, each temporal modality (climate / vegetation / soil
moisture) is given a small autoregressive forecaster: predict step t+1 from steps
0..t. Minimizing next-step error forces the temporal encoders to model dynamics
(seasonality, wetting/drying, phenology), so the fused latent carries change.

Wired into GeoSoilModel (self.forecasters) and losses.compute_losses (w_forecast).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from . import config as C

# modality name -> per-step channel count
TEMPORAL_MODALITIES = {
    "climate": len(C.CLIMATE_VARS),
    "veg": len(C.VEG_VARS),
    "moisture": len(C.MOISTURE_VARS),
}


class SequenceForecaster(nn.Module):
    """Autoregressive next-step predictor for one temporal modality."""

    def __init__(self, n_vars: int, hidden: int = 48):
        super().__init__()
        self.gru = nn.GRU(n_vars, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_vars)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:        # (B,L,V)
        out, _ = self.gru(seq[:, :-1])                           # predict from prefix
        return self.head(out)                                    # (B,L-1,V) = forecast of seq[:,1:]


def build_forecasters(names: List[str]) -> nn.ModuleDict:
    return nn.ModuleDict({
        n: SequenceForecaster(v) for n, v in TEMPORAL_MODALITIES.items() if n in names
    })


def forecast_loss(forecasters: nn.ModuleDict, batch: dict, presence_names: List[str]) -> torch.Tensor:
    """Mean next-step MSE over present temporal modalities (present rows only)."""
    total, k = None, 0
    for name, fc in forecasters.items():
        seq = batch.get(name)
        if seq is None or seq.dim() != 3:
            continue
        pres = batch.get(f"{name}_present")
        pred = fc(seq)
        target = seq[:, 1:]
        err = ((pred - target) ** 2).mean(dim=(1, 2))           # (B,)
        if pres is not None:
            w = pres.squeeze(-1)
            loss = (err * w).sum() / w.sum().clamp(min=1.0)
        else:
            loss = err.mean()
        total = loss if total is None else total + loss
        k += 1
    if total is None:
        # no temporal modality present -> zero (keeps graph valid)
        return next(iter(forecasters.parameters())).sum() * 0.0 if len(forecasters) else torch.zeros(())
    return total / k
