"""Temporal sequence encoders for the climate / vegetation 12-month modalities.

Three interchangeable encoders for the bake-off:
  - GRU          : strong, cheap recurrent baseline
  - Transformer  : self-attention over the 12 monthly steps
  - Mamba (S6)   : a minimal *pure-PyTorch* selective state-space block. The
                   official mamba-ssm CUDA kernel does not run on Mac MPS/CPU,
                   so we implement the selective-scan recurrence directly. For
                   length-12 sequences the sequential scan is perfectly fast.

All encoders map (B, L, n_vars) -> (B, out_dim).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, n_vars: int, hidden: int, out_dim: int):
        super().__init__()
        self.gru = nn.GRU(n_vars, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,V)
        _, h = self.gru(x)
        return self.proj(h[-1])


class TransformerTemporal(nn.Module):
    def __init__(self, n_vars: int, hidden: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.inp = nn.Linear(n_vars, hidden)
        pe = torch.zeros(64, hidden)
        pos = torch.arange(64).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, hidden, 2).float() * (-math.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,V)
        h = self.inp(x) + self.pe[: x.size(1)].unsqueeze(0)
        h = self.enc(h)
        return self.proj(h.mean(dim=1))


class SelectiveSSM(nn.Module):
    """Minimal Mamba-style selective state-space block (diagonal A, S6 selection).

    Per step the discretization step-size `delta`, input matrix `B`, and output
    matrix `C` are all functions of the input (selective). The state recurrence
    h_t = exp(delta_t * A) h_{t-1} + (delta_t * B_t) x_t ;  y_t = C_t h_t  is run
    as an explicit scan over the (short) sequence.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        # Learnable diagonal state matrix A (kept negative via -exp).
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_model, 1))
        self.D = nn.Parameter(torch.ones(d_model))
        # Input-dependent projections for delta, B, C (the "selection").
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,D)
        B, L, D = x.shape
        A = -torch.exp(self.A_log)                                  # (D, N)
        proj = self.x_proj(x)                                       # (B,L,2N+1)
        dt, Bm, Cm = proj[..., :1], proj[..., 1:1 + self.d_state], proj[..., 1 + self.d_state:]
        delta = F.softplus(self.dt_proj(dt))                        # (B,L,D)
        h = x.new_zeros(B, D, self.d_state)
        ys = []
        for t in range(L):
            dA = torch.exp(delta[:, t].unsqueeze(-1) * A)           # (B,D,N)
            dBx = delta[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
            h = dA * h + dBx                                        # (B,D,N)
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))         # (B,D)
        y = torch.stack(ys, dim=1) + x * self.D                    # (B,L,D)
        return y


class MambaTemporal(nn.Module):
    """Mamba temporal encoder. Uses the official `mamba-ssm` CUDA selective-scan
    kernel when available (fast, on Alpine GPUs); falls back to the pure-PyTorch
    SelectiveSSM otherwise (correct but slow — fine on CPU/MPS for smoke tests)."""

    def __init__(self, n_vars: int, hidden: int, out_dim: int, d_state: int = 16):
        super().__init__()
        self.inp = nn.Linear(n_vars, hidden)
        self.norm = nn.LayerNorm(hidden)
        try:
            from mamba_ssm import Mamba                  # CUDA kernel
            self.block = Mamba(d_model=hidden, d_state=d_state, d_conv=4, expand=2)
            self.kernel = "mamba-ssm"
            self.gate = None
        except Exception:
            self.block = SelectiveSSM(hidden, d_state)    # pure-PyTorch fallback
            self.gate = nn.Linear(hidden, hidden)
            self.kernel = "pytorch"
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,L,V)
        h = self.inp(x)
        if self.gate is None:                             # real Mamba block (has its own residual/gating)
            h = h + self.block(self.norm(h))
        else:                                             # gated SelectiveSSM residual
            h = h + self.block(self.norm(h)) * torch.sigmoid(self.gate(h))
        return self.proj(h.mean(dim=1))


def make_temporal_encoder(kind: str, n_vars: int, hidden: int, out_dim: int) -> nn.Module:
    kind = kind.lower()
    if kind == "gru":
        return GRUEncoder(n_vars, hidden, out_dim)
    if kind == "transformer":
        return TransformerTemporal(n_vars, hidden, out_dim)
    if kind == "mamba":
        return MambaTemporal(n_vars, hidden, out_dim)
    raise ValueError(f"unknown temporal encoder: {kind}")
