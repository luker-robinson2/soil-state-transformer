"""GeoSoil model: encode modalities -> attention fusion -> high-dim latent z,
with heads for multi-target heteroscedastic regression and self-supervised
objectives (cross-modal JEPA, InfoNCE, masked modeling, VICReg).
"""
from __future__ import annotations

import copy
from typing import Dict, List

import torch
import torch.nn as nn

from . import config as C
from .forecasting import build_forecasters, forecast_loss
from .modalities import ModalityEncoders


class GeoSoilModel(nn.Module):
    def __init__(self, cfg: C.GeoSoilConfig, has: dict, n_targets: int = len(C.TARGETS)):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.encoders = ModalityEncoders(cfg, has)
        self.names: List[str] = self.encoders.names
        d, L = m.modality_dim, m.latent_dim

        # project each modality token into the fusion/latent width
        self.to_latent = nn.Linear(d, L)
        self.cls = nn.Parameter(torch.randn(1, 1, L) * 0.02)
        layer = nn.TransformerEncoderLayer(L, m.fusion_heads, L * 2, batch_first=True, dropout=m.dropout)
        # enable_nested_tensor=False: the nested-tensor fast path lacks an MPS kernel
        self.fusion = nn.TransformerEncoder(layer, num_layers=m.fusion_layers, enable_nested_tensor=False)

        # heads
        self.head = nn.Sequential(
            nn.Linear(L, m.head_hidden), nn.GELU(), nn.Dropout(m.dropout),
            nn.Linear(m.head_hidden, n_targets * 2),         # mu + log_var per target
        )
        proj_dim = m.modality_dim
        self.token_proj = nn.Sequential(nn.Linear(d, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.predictor = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))
        self.recon = nn.Linear(L, len(C.AE_COLS))            # masked AE reconstruction
        self.vic_proj = nn.Sequential(nn.Linear(L, L), nn.GELU(), nn.Linear(L, L))
        self.forecasters = build_forecasters(self.names)     # temporal dynamics objective

        # EMA target copy of (encoders + token_proj) for JEPA targets
        self.t_encoders = copy.deepcopy(self.encoders)
        self.t_token_proj = copy.deepcopy(self.token_proj)
        for p in list(self.t_encoders.parameters()) + list(self.t_token_proj.parameters()):
            p.requires_grad_(False)

    # ------------------------------------------------------------------ #
    def fuse(self, tokens: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        B = tokens.size(0)
        h = self.to_latent(tokens)                                   # (B,M,L)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)                               # (B,M+1,L)
        pad = torch.cat([torch.ones(B, 1, device=presence.device), presence], dim=1)
        z = self.fusion(h, src_key_padding_mask=(pad < 0.5))         # CLS attends to present
        return z[:, 0]                                               # (B,L)

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        tokens, presence = self.encoders(batch)                     # (B,M,d),(B,M)
        z = self.fuse(tokens, presence)
        out = self.head(z)
        n = out.size(-1) // 2
        return {
            "z": z, "tokens": tokens, "presence": presence,
            "proj": self.token_proj(tokens),                        # (B,M,proj)
            "mu": out[:, :n], "log_var": out[:, n:].clamp(-8, 8),
            "vic": self.vic_proj(z),
        }

    @torch.no_grad()
    def target_proj(self, batch: dict):
        tokens, presence = self.t_encoders(batch)
        return self.t_token_proj(tokens), presence                  # (B,M,proj),(B,M)

    def predict_cross(self, proj: torch.Tensor) -> torch.Tensor:
        return self.predictor(proj)

    def recon_ae(self, batch_masked: dict) -> torch.Tensor:
        tokens, presence = self.encoders(batch_masked)
        return self.recon(self.fuse(tokens, presence))

    def forecast(self, batch: dict) -> torch.Tensor:
        return forecast_loss(self.forecasters, batch, self.names)

    @torch.no_grad()
    def ema_update(self, decay: float):
        for o, t in zip(self.encoders.parameters(), self.t_encoders.parameters()):
            t.mul_(decay).add_(o.detach(), alpha=1 - decay)
        for o, t in zip(self.token_proj.parameters(), self.t_token_proj.parameters()):
            t.mul_(decay).add_(o.detach(), alpha=1 - decay)
