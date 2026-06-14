"""GeoSoil configuration: canonical targets, modalities, paths, hyperparameters.

A personal-research evolution of the Soil State Transformer into a multi-modal,
multi-truth geospatial soil representation model. All settings are plain
dataclasses so a run is fully described by one object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# --- Repo-relative paths -----------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
DATA_RAW = REPO / "alphaearth-soc" / "future_work" / "data"
DATA_ORIG = REPO / "alphaearth-soc" / "data"
DATA_PROC = REPO / "data" / "processed"
RESULTS = Path(__file__).resolve().parent / "results"

# --- Canonical schema --------------------------------------------------------
# The 5 harmonized soil targets, in canonical units, used everywhere downstream.
TARGETS: Tuple[str, ...] = ("soc", "ph", "sand", "clay", "bd")
TARGET_UNITS = {"soc": "g/kg", "ph": "pH(H2O)", "sand": "%", "clay": "%", "bd": "g/cm3"}
# log1p transform applied to these targets before modeling (right-skewed).
LOG_TARGETS: Tuple[str, ...] = ("soc",)

# AlphaEarth embedding columns A00..A63 (always present).
AE_COLS: Tuple[str, ...] = tuple(f"A{i:02d}" for i in range(64))
# Sentinel-2 static spectral columns (present for the paired subset).
S2_COLS: Tuple[str, ...] = ("B2", "B4", "B6", "B8", "B11", "B12", "NDVI", "EVI", "NDWI", "NBR")
# Terrain/DEM columns (filled by GEE extraction; absent -> masked).
TERRAIN_COLS: Tuple[str, ...] = ("elevation", "slope", "aspect_sin", "aspect_cos", "twi")
# Soil-sensing remote modalities (extract_gee_soil.py): bare-soil S2 + Sentinel-1 SAR.
BARESOIL_COLS: Tuple[str, ...] = ("BS_B2", "BS_B4", "BS_B6", "BS_B8", "BS_B11", "BS_B12")
SAR_COLS: Tuple[str, ...] = ("S1_VV", "S1_VH", "S1_ratio")

# Temporal modalities (12-month sequences) filled by GEE extraction.
# Each is (name, per-step channel columns prefix); absent -> masked & branch skipped.
CLIMATE_VARS: Tuple[str, ...] = ("t2m", "tprate", "stl1", "swvl1")   # ERA5-Land monthly
VEG_VARS: Tuple[str, ...] = ("ndvi", "evi")                           # MODIS monthly
MOISTURE_VARS: Tuple[str, ...] = ("sm",)                              # SMAP soil moisture monthly
SEQ_LEN: int = 12

# Micro-event / field-alteration static features (extract_gee_dynamics.py).
PRECIP_COLS: Tuple[str, ...] = ("precip_total", "precip_wetdays", "precip_max1d", "precip_p95")
CDL_COLS: Tuple[str, ...] = ("cdl_n_crops", "cdl_changes", "cdl_frac_corn", "cdl_frac_soy", "cdl_frac_wwheat")


@dataclass
class ModelConfig:
    latent_dim: int = 256          # high-dim shared latent z
    modality_dim: int = 128        # per-modality embedding width
    geo_num_freqs: int = 16        # Fourier features for (lon,lat)
    geo_sigma: float = 8.0
    fusion_heads: int = 4
    fusion_layers: int = 2
    dropout: float = 0.1
    temporal_encoder: str = "gru"  # one of: gru | transformer | mamba
    temporal_hidden: int = 64
    head_hidden: int = 128


@dataclass
class LossConfig:
    w_supervised: float = 1.0      # multi-target heteroscedastic Gaussian NLL
    w_jepa: float = 0.5            # cross-modal predict-in-latent (EMA target)
    w_infonce: float = 0.3        # cross-modal contrastive (AE<->S2 / AE<->temporal)
    w_mask: float = 0.3           # masked feature modeling
    w_vicreg: float = 0.04        # variance/covariance collapse prevention
    w_forecast: float = 0.2       # temporal next-step forecasting (dynamics awareness)
    infonce_temp: float = 0.1
    mask_ratio: float = 0.30
    ema_decay: float = 0.996      # target-encoder EMA for JEPA


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    n_spatial_folds: int = 5
    block_deg: float = 1.0         # 1deg x 1deg spatial blocks for GroupKFold
    ensemble_seeds: Tuple[int, ...] = (0, 1)
    conformal_alpha: float = 0.1   # 90% prediction intervals
    device: str = "auto"           # auto -> cuda > mps > cpu
    seed: int = 0
    # --- GPU / scaling knobs (Alpine) ---
    precision: str = "fp32"        # fp32 | bf16 | fp16  (AMP autocast on cuda)
    compile: bool = False          # torch.compile the model
    num_workers: int = 0           # DataLoader workers (set >0 on cuda)
    ddp: bool = False              # DistributedDataParallel (torchrun) for the big pretrain


# --- Model size presets for the scaling study (size vs performance) ---
MODEL_PRESETS = {
    "small": dict(latent_dim=128, modality_dim=64, fusion_heads=4, fusion_layers=2,
                  temporal_hidden=48, head_hidden=64),
    "base":  dict(latent_dim=256, modality_dim=128, fusion_heads=4, fusion_layers=2,
                  temporal_hidden=64, head_hidden=128),   # current default
    "large": dict(latent_dim=512, modality_dim=256, fusion_heads=8, fusion_layers=4,
                  temporal_hidden=128, head_hidden=256),
    "xl":    dict(latent_dim=768, modality_dim=384, fusion_heads=12, fusion_layers=6,
                  temporal_hidden=192, head_hidden=384),
}


def model_preset(name: str = "base") -> "ModelConfig":
    return ModelConfig(**MODEL_PRESETS[name])


@dataclass
class GeoSoilConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def preset(cls, size: str = "base") -> "GeoSoilConfig":
        cfg = cls()
        cfg.model = model_preset(size)
        return cfg


def resolve_device(name: str = "auto") -> str:
    import torch
    if name != "auto":
        return name
    if torch.cuda.is_available():          # prefer GPU (Alpine) when present
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
