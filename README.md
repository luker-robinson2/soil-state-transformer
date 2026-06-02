# Soil Machine Learning — Portfolio

Foundation-model and geospatial machine learning for **soil property prediction**.
This repo holds the data-science work behind my portfolio site, in two parts that
together tell one story: *can modern representation learning predict what's in the
soil from above?*

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
├── alphaearth-soc/      # AlphaEarth→SOC study (paper, R, GEE, data)
├── foundation-model/    # Soil State Transformer research
├── docs/
│   ├── PROVENANCE.md    # where each piece came from
│   └── SCRUB_LOG.md     # record of the SST sanitization pass
└── README.md
```

## Roadmap toward the public artifact
The highest-ROI next step is **one polished public artifact**: an SST write-up or
self-contained notebook built around the WoSIS-only pre-training story. The
AlphaEarth/SOC study is publishable as-is and is the natural front door for the
portfolio site.

## Tech
Python · PyTorch · scikit-learn · R · Google Earth Engine · WoSIS · ERA5 ·
SoilGrids · Vertex AI · LoRA
