"""Per-modality encoders that map each input modality into the shared latent.

FourierFeatures is reused from the Soil State Transformer lineage
(foundation-model/models/soil_state_transformer.py) — geographic coordinates are
lifted to multi-scale sinusoids before the MLP (Tancik et al., NeRF-style).
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from . import config as C
from .temporal import make_temporal_encoder


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int = 2, num_freqs: int = 16, sigma: float = 8.0):
        super().__init__()
        self.register_buffer("B", torch.randn(input_dim, num_freqs) * sigma)
        self.output_dim = num_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModalityEncoders(nn.Module):
    """Builds one encoder per *available* modality; emits a list of tokens.

    Each modality -> a (B, modality_dim) token. A presence mask zeroes absent
    modalities so the fusion attention can ignore them.
    """

    def __init__(self, cfg: C.GeoSoilConfig, has: dict):
        super().__init__()
        m = cfg.model
        d = m.modality_dim
        self.has = has
        self.names: List[str] = ["ae", "geo"]
        self.enc = nn.ModuleDict()
        self.enc["ae"] = MLPEncoder(len(C.AE_COLS), d, dropout=m.dropout)
        self.geo_fourier = FourierFeatures(2, m.geo_num_freqs, m.geo_sigma)
        self.enc["geo"] = MLPEncoder(self.geo_fourier.output_dim, d, dropout=m.dropout)
        if has.get("s2"):
            self.enc["s2"] = MLPEncoder(len(C.S2_COLS), d, dropout=m.dropout); self.names.append("s2")
        if has.get("terrain"):
            self.enc["terrain"] = MLPEncoder(len(C.TERRAIN_COLS), d, dropout=m.dropout); self.names.append("terrain")
        if has.get("baresoil"):
            self.enc["baresoil"] = MLPEncoder(len(C.BARESOIL_COLS), d, dropout=m.dropout); self.names.append("baresoil")
        if has.get("sar"):
            self.enc["sar"] = MLPEncoder(len(C.SAR_COLS), d, dropout=m.dropout); self.names.append("sar")
        if has.get("precip"):
            self.enc["precip"] = MLPEncoder(len(C.PRECIP_COLS), d, dropout=m.dropout); self.names.append("precip")
        if has.get("cdl"):
            self.enc["cdl"] = MLPEncoder(len(C.CDL_COLS), d, dropout=m.dropout); self.names.append("cdl")
        if has.get("climate"):
            self.enc["climate"] = make_temporal_encoder(m.temporal_encoder, len(C.CLIMATE_VARS), m.temporal_hidden, d)
            self.names.append("climate")
        if has.get("veg"):
            self.enc["veg"] = make_temporal_encoder(m.temporal_encoder, len(C.VEG_VARS), m.temporal_hidden, d)
            self.names.append("veg")
        if has.get("moisture"):
            self.enc["moisture"] = make_temporal_encoder(m.temporal_encoder, len(C.MOISTURE_VARS), m.temporal_hidden, d)
            self.names.append("moisture")

    def forward(self, batch: dict):
        tokens, presence = [], []
        B = batch["ae"].shape[0]
        for name in self.names:
            if name == "geo":
                tok = self.enc["geo"](self.geo_fourier(batch["geo"]))
                pres = torch.ones(B, 1, device=tok.device)
            else:
                tok = self.enc[name](batch[name])
                pres = batch.get(f"{name}_present", torch.ones(B, 1, device=tok.device))
            tokens.append(tok * pres)        # zero absent modalities
            presence.append(pres)
        toks = torch.stack(tokens, dim=1)            # (B, M, d)
        pres = torch.cat(presence, dim=1)            # (B, M)
        return toks, pres
