# Soil Machine Learning — Portfolio

Foundation-model and geospatial machine learning for **soil property prediction**.
This repo holds the data-science work behind my portfolio site. One story, three
parts: *can modern representation learning predict what's in the soil from above —
and produce a reusable latent map of place?*

---

## ⭐ GeoSoil — multi-modal geospatial soil representation  ·  [`geosoil/`](geosoil/)
**The flagship.** A high-dimensional **learned latent** of soil + geospatial state,
fused from AlphaEarth embeddings, Sentinel-2 spectra, geography, terrain, and
climate/vegetation time series, trained with **self-supervised cross-modal
objectives** (JEPA + InfoNCE + masked modeling + VICReg) and a multi-target
heteroscedastic head. Verified against **lab-measured KSSL truth** under
spatial-block CV, with a **bake-off** of JEPA / Mamba / EBM variants vs strong
tabular baselines (LightGBM/XGBoost/CatBoost/TabPFN) and a frozen-latent→GBM
hybrid. The evolution of the Soil State Transformer below.

---

## 1. AlphaEarth → Soil Organic Carbon  ·  [`alphaearth-soc/`](alphaearth-soc/)
**A complete, reproducible study on 100% public data.** ✅ *Showcase-ready.*

Do Google DeepMind's **AlphaEarth Foundations** 64-dim geospatial embeddings carry
usable signal for **Soil Organic Carbon (SOC)** — even though they were never
trained on any soil target?

- **Data:** n = 3,000 random CONUS points; AlphaEarth 2024 embeddings ×
  OpenLandMap SOC × MODIS land cover, all via Google Earth Engine.
- **Methods:** exploratory data analysis · Welch's two-sample test ·
  nonparametric **bootstrap (BCa) confidence intervals** · spatial-block
  cross-validated Random Forest.
- **Headline:** several embedding dims correlate with log-SOC up to |r| = 0.61;
  cropland vs grassland log-SOC differs significantly; spatial-CV RF reaches
  **R² = 0.75** out-of-fold.
- **Artifacts:** full LaTeX paper (`paper/main.pdf`), the R analysis
  (`analysis/soil_analysis.Rmd`), the GEE extraction script, figures, and the
  `soil_samples.csv` dataset.
- **`future_work/`** extends this to multi-target prediction (pH, sand, clay, BD),
  a Sentinel-2-only baseline, and KSSL lab ground-truth validation.

## 2. Soil State Transformer (SST)  ·  [`foundation-model/`](foundation-model/)
**A transformer foundation model for soil nutrient dynamics.** Personal research,
built on public data.

Pre-train on global soil data, then **LoRA-fine-tune** on local fields.

- **Architecture:** multi-modal transformer — Fourier geographic embeddings,
  learned depth embeddings, per-property soil embeddings with learnable
  missing-value masks, hierarchical temporal (weather) encoding, 6-layer /
  8-head encoder, prediction heads with uncertainty.
- **Pre-training:** masked soil-property prediction on **WoSIS** (~230k global
  profiles, 79k-sample test set). Test-set **R²: sand 0.83, silt 0.70,
  clay 0.49**, overall 0.41 — a clean, honest result on a hard global task.
- **Transfer:** LoRA adapters (~90% parameter reduction) for field-level
  fine-tuning; cloud training on Vertex AI with spot GPUs.
- **Code:** model, training loop, datasets, evaluation, notebooks (`01`–`10`).

> **Independent work.** This is a personal research project of mine, built and
> trained entirely on public datasets (WoSIS, ERA5, SoilGrids). It contains no
> proprietary data or infrastructure. Config values like GCP project / bucket are
> placeholders (`your-gcp-project`, `your-research-bucket`) — set your own.

---

## Repository layout

```
.
├── geosoil/             # ⭐ flagship: multi-modal latent soil representation model
├── alphaearth-soc/      # AlphaEarth→SOC study (paper, R, GEE, data)
├── foundation-model/    # Soil State Transformer research (GeoSoil's predecessor)
├── data/                # Phase-0 extraction (GEE covariates) + processed master table
├── docs/
│   ├── research/        # SOTA survey behind GeoSoil's design
│   ├── PROVENANCE.md    # where each piece came from
│   └── SCRUB_LOG.md     # record of the SST sanitization pass
└── README.md
```

## Tech
Python · PyTorch (MPS) · scikit-learn · LightGBM/XGBoost/CatBoost/TabPFN · R ·
Google Earth Engine · AlphaEarth · Sentinel-2 · WoSIS · KSSL · ERA5 · SoilGrids ·
JEPA · contrastive SSL · Mamba/SSM · energy-based models · conformal prediction
