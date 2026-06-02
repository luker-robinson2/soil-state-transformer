# Soil State Transformer — Claude Code Guidance

Personal research project: a transformer foundation model for soil property
prediction, built on public datasets (WoSIS, ERA5, SoilGrids). Notebook-first.

## Development approach

1. Prototype experiments in `notebooks/` (01–04 data, 05–08 modeling, 09–10 eval).
2. Promote validated code into `models/`, `training/`, `data/` modules.
3. Keep notebooks as the running record of what was tried.

## Directory structure

```
foundation-model/
├── notebooks/                 # PRIMARY - start here
│   ├── data_exploration/      # 01-04: understand the data
│   ├── model_prototyping/     # 05-08: design models
│   └── evaluation/            # 09-10: validate results
├── data/                      # data downloaders (WoSIS / ERA5 / SoilGrids)
├── models/                    # SST architecture, pre-train head, LoRA wrapper
├── training/                  # training loop, datasets, metrics, config
├── results/                   # experiment outputs
└── scripts/                   # prepare data, cloud jobs, evaluation
```

## Model architecture

```
Soil State Transformer (SST)
├── Geographic embedding    — Fourier features over (lat, lon)
├── Depth embedding         — learned depth representation
├── Soil-property embedding — per-property + learnable missing-value mask
├── Temporal embedding      — hierarchical weather-history encoding
├── Transformer encoder     — 6 layers, 8 heads
└── Prediction heads        — property prediction + uncertainty (Softplus)
```

## Data sources (all public)

| Source | Description | Access |
|--------|-------------|--------|
| WoSIS | Global soil profiles | ISRIC REST API |
| ERA5 | Climate reanalysis | Copernicus CDS API (requires free key) |
| SoilGrids | 250 m global soil predictions | Direct download |

## Common tasks

```bash
# Download data
python -m data.wosis_downloader --bbox=-125,24,-66,50   # CONUS
python -m data.era5_downloader --years=2020,2024

# Pre-train locally
python -m training.pretrain \
    --train-path data/processed/wosis_train.parquet \
    --val-path   data/processed/wosis_val.parquet

# Evaluate
python scripts/evaluate_pretrain.py --test-data data/processed/wosis_test.parquet

# Run a notebook end-to-end as a smoke test
jupyter nbconvert --execute --to notebook notebooks/data_exploration/01_wosis_exploration.ipynb
```

## Config

Central config in `config.yml` (model dims, data sources, paths, GCP). Code follows
a dataclass config pattern (see `training/config.py`). GCP `project_id` /
`research_bucket` are placeholders — set your own, or run fully local (GCS is only
used by the cloud-training scripts).

## Experiment tracking

Optional Weights & Biases:

```python
import wandb
wandb.init(project='soil-state-transformer')
wandb.log({'train_loss': loss, 'val_r2': r2})
```

## Cloud training (optional, Vertex AI)

```bash
python scripts/upload_data_to_gcs.py
./scripts/build_training_container.sh
python scripts/submit_training_job.py --gpu-type v100 --use-spot
python scripts/monitor_training_job.py --job-id <job-id>
python scripts/download_model.py --wandb-run-id <run-id>
```

Checkpoints organize under `gs://<your-research-bucket>/checkpoints/pretrain/<run-id>/`.
Use spot/preemptible instances; checkpoint every ~30 min so jobs auto-resume.
