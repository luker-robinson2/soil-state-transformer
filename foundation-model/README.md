# Soil State Transformer (SST)

A transformer **foundation model for soil property prediction** — pre-trained on
global soil data, then adapted to local fields with LoRA. A personal research
project exploring whether a single multi-modal sequence model can learn a useful
representation of "soil state" from location, depth, climate history, and a few
base properties.

## Idea

Most soil-property models are tabular regressors trained per-property on one
region. The SST instead treats a soil observation as a **sequence of modalities**
and learns a shared representation via masked pre-training on a large, public
global database — the same recipe that works for language and vision foundation
models.

- **Pre-train** on global soil profiles with **masked property prediction**
  (mask 15% of inputs, reconstruct them).
- **Transfer** to a specific field with **LoRA** adapters (~90% fewer trainable
  parameters), so a few hundred local samples are enough to specialize.

## Architecture

```
Soil State Transformer
├── Geographic embedding   — Fourier features over (lat, lon)
├── Depth embedding        — learned, from upper/lower/thickness/center depth
├── Soil-property embedding — per-property projection + learnable missing-value mask
├── Temporal embedding     — hierarchical encoding of weather history (weekly/monthly)
├── Transformer encoder    — 6 layers, 8 heads, self-attention over modalities
└── Prediction heads       — property prediction + uncertainty (Softplus)
```

See `models/soil_state_transformer.py` (architecture), `models/sst_pretrain.py`
(masked pre-training head), and `models/lora_wrapper.py` (LoRA adapters).

## Results so far

Pre-trained on **WoSIS** (ISRIC's public ~230k global soil profiles), evaluated on
a held-out 79k-sample test set with masked-property reconstruction:

| Property | R² | Notes |
|----------|-----|-------|
| sand | 0.83 | learned texture relationships well |
| silt | 0.70 | |
| clay | 0.49 | |
| bulk density | 0.41 | |
| organic carbon | 0.38 | |
| CEC | 0.37 | |
| nitrogen | 0.32 | |
| pH | −0.12 | hardest target; honest negative result |

Overall R² ≈ 0.41. Texture is learnable from context; pH is not (yet) — a useful
signal about what the global features do and don't carry. Full numbers in
`results/pretrain_evaluation/`.

## Data sources (all public)

| Source | What | Access |
|--------|------|--------|
| **WoSIS** | Global soil profiles | ISRIC REST API (`data/wosis_downloader.py`) |
| **ERA5** | Climate reanalysis | Copernicus CDS API (`data/era5_downloader.py`) |
| **SoilGrids** | 250 m global soil covariates | Direct download (`data/soilgrids_downloader.py`) |

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download + prepare WoSIS pre-training data
python -m data.wosis_downloader --bbox=-125,24,-66,50   # CONUS example
python scripts/prepare_wosis.py

# Pre-train locally
python -m training.pretrain \
    --train-path data/processed/wosis_train.parquet \
    --val-path   data/processed/wosis_val.parquet

# Evaluate
python scripts/evaluate_pretrain.py --test-data data/processed/wosis_test.parquet
```

Cloud (Vertex AI) training scripts live in `scripts/` and read GCP project / bucket
from `config.yml` — set those to your own, or skip them and run locally.

## Layout

```
foundation-model/
├── models/        SST architecture, pre-training head, LoRA wrapper
├── training/      training loop, datasets, metrics, config, GCS + Vertex helpers
├── data/          downloaders for WoSIS / ERA5 / SoilGrids
├── scripts/       prepare data, submit/monitor cloud jobs, evaluate
├── notebooks/     01–04 data exploration · 05–08 model prototyping · 09–10 eval
├── results/       pre-training evaluation outputs
├── config.yml     central config (model, data, paths, GCP)
└── requirements.txt
```

## Roadmap / research directions

- Add ERA5 temporal features and test the temporal-prediction pre-training task.
- Multi-target fine-tuning and uncertainty calibration.
- Compare the learned representation against a plain XGBoost/RF baseline
  (notebooks `05`–`10`).
- Tie in with the [AlphaEarth → SOC study](../alphaearth-soc/) — can AlphaEarth
  embeddings serve as an additional input modality?

## Notes

This is independent personal research built on public datasets. No proprietary
data or infrastructure. GCP project/bucket names in config and scripts are
placeholders (`your-gcp-project`, `your-research-bucket`) — set your own.
