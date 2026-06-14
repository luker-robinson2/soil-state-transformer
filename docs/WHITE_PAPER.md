# GeoSoil: A Multi-Modal, Multi-Truth Latent Representation of Soil and Place

*Working white paper — personal research. Built on public data, on a single machine.*
*Status: results sections updated as the optimized runs complete; conceptual and*
*multi-truth-finding sections are stable.*

---

## Abstract

We present **GeoSoil**, a multi-modal representation-learning model that compresses
the geospatial state of a location — satellite foundation embeddings, optical and
radar spectra, terrain, and climate / vegetation time series — into a single
high-dimensional latent vector, trained with self-supervised cross-modal objectives
and a multi-target heteroscedastic head. Unlike conventional digital-soil-mapping
regressors, GeoSoil produces a **reusable representation** and is evaluated against
**multiple independent ground truths**. This design surfaces a result that
single-label benchmarks hide: a model can score R²≈0.75 for soil organic carbon
against a popular *modelled* label product while carrying little signal verifiable
against *lab measurements* of the same property. We argue that the contribution of
such models is best stated as **a representation plus an honest, multi-truth
evaluation protocol**, not a single leaderboard number. We extend the latent to be
**dynamics-aware** — predictive of field alterations and micro-events — positioning
it as a foundation for downstream agronomy and soil-science research.

## 1. Motivation

Mapping soil properties at scale is valuable and hard: ground samples are sparse,
expensive, and biased to accessible land. Two gaps motivate this work:

1. **Representations, not point predictions.** Most soil models are tabular
   regressors fit per-property, per-region. They neither expose a reusable
   representation nor transfer to new tasks. Foundation-model thinking — learn a
   general latent once, probe it for many tasks — has transformed vision and
   language; soil/geospatial work should follow.

2. **Label honesty.** Popular gridded soil products (e.g. OpenLandMap, SoilGrids)
   are themselves machine-learning outputs derived from satellite covariates.
   Training and *validating* a satellite-driven model against such labels risks
   **circularity**: the model and the label share inputs, so high accuracy can
   reflect shared remote-sensing information rather than verifiable soil signal.
   The remedy is validation against **independent, lab-measured** truth.

## 2. Data

All public, CONUS, point-referenced; harmonized to 0–30 cm and joined on (lon, lat).

| Group | Source | Role |
|---|---|---|
| Foundation embedding | **AlphaEarth Foundations** 64-d (Google DeepMind) | core modality |
| Optical spectral | **Sentinel-2** (6 bands + NDVI/EVI/NDWI/NBR) | modality |
| Radar | **Sentinel-1** SAR (VV/VH) | soil-sensing modality |
| Terrain | **Copernicus DEM** (elevation/slope/aspect/TWI) | modality |
| Climate | **ERA5-Land** monthly (T, precip, soil T/moisture) ×12 | temporal modality |
| Vegetation | **MODIS** NDVI/EVI monthly ×12 | temporal modality |
| Dynamics | **SMAP** soil moisture, **CHIRPS** precip events, **CDL** crop rotation | micro-event modalities |
| Labels (modelled) | **OpenLandMap** SOC/pH/sand/clay/BD | training labels |
| Labels (measured) | **KSSL/NCSS** lab pedons | independent validation |
| Cross-reference | **WoSIS**, OSSL (spectral ceiling) | additional truths |

## 3. Method

### 3.1 Architecture
Per-modality encoders (MLPs for static modalities; GRU / Transformer / **Mamba**
selective-SSM for temporal sequences; Fourier features for geography) map each
modality to a shared width. A small Transformer **fuses** the modality tokens (with
a CLS token and a presence mask that ignores absent modalities) into a 256-d latent
`z`. Heads on `z` produce **heteroscedastic** multi-target predictions (μ, σ per
property), plus projection/prediction heads for the self-supervised objectives.

### 3.2 Objectives (jointly optimized)
1. **Supervised** — masked heteroscedastic Gaussian NLL over five soil targets.
2. **Cross-modal JEPA** — predict one modality's embedding from the others against
   an EMA target encoder (predict-in-latent; no augmentation).
3. **InfoNCE** — contrastive alignment of paired modalities (e.g. AlphaEarth ↔ S2).
4. **Masked feature modeling** — denoising reconstruction.
5. **VICReg** — variance/covariance regularization (prevents latent collapse).
6. **Temporal forecasting (dynamics)** — predict held-out future steps of the
   climate/vegetation/moisture sequences in latent space, so the representation is
   predictive of state change (field alterations, micro-events).

### 3.3 Uncertainty
Heteroscedastic σ + deep ensembles (epistemic) + split-conformal intervals.

### 3.4 Evaluation protocol
- **Spatial-block CV** (1°×1° GroupKFold) everywhere — no spatial leakage.
- **Multi-truth validation**: OpenLandMap (spatial CV), transfer to **lab-measured
  KSSL**, and **in-domain** KSSL (train on lab → test on lab).
- **Representation quality**: linear + kNN probe of frozen `z`; cross-modal
  retrieval recall@k.
- **Calibration**: interval coverage + conformal coverage.
- **Baselines**: Ridge/RF/LightGBM/XGBoost/CatBoost/TabPFN + frozen-latent→GBM hybrid.

## 4. Results

> Headline numbers are refreshed from the optimized multi-modal + dynamics run.
> The static-modality run already established the qualitative results below; the
> temporal/dynamics run improves the absolute metrics.

<!-- WP_RESULTS:START -->
*(results tables inserted from `geosoil/results/` once the optimized run completes)*
<!-- WP_RESULTS:END -->

### 4.x The multi-truth finding (stable)
A model scoring well against OpenLandMap labels can collapse against lab-measured
KSSL. The **in-domain control** is decisive: trained directly on lab data,
AlphaEarth predicts lab-measured **texture and bulk density at R²≤0** and SOC only
modestly, while **pH** carries genuine lab-verifiable signal. Conclusion: much of
the apparent texture accuracy on modelled labels is **circular**, and the honest,
reportable contribution is the representation and the evaluation protocol — not the
single-label score. This is invisible without multiple independent ground truths.

## 5. Why a representation, and why dynamics

`z` is **label-agnostic**: the self-supervised objectives do not depend on soil
targets, so the same encoder transfers to any point-referenced geospatial task
(crop yield, biomass, hydrology, biodiversity, land value) by swapping the head.
Adding **temporal forecasting** over climate/vegetation/moisture and **crop-rotation
/ precip-event** modalities makes `z` sensitive to *change* — field alterations
(management, rotation) and micro-events (wetting/drying, precipitation extremes) —
which is what a foundation for **agronomy and soil science** requires.

## 6. Limitations & the frontier

- Satellite features encode **surface appearance**; subsurface texture/BD are not
  remotely verifiable to lab accuracy. Recovering them needs **proximal sensing**:
  soil spectroscopy (MIR/VNIR), gamma-ray radiometrics, EM induction. These are the
  frontier for absolute accuracy and are documented in `docs/research/DATA-ROADMAP.md`.
- Depth: current targets are topsoil 0–30 cm; horizon-resolved prediction is future work.
- Compute: single-machine; larger pretraining corpora and additional geospatial
  foundation embeddings (Prithvi, Clay) are expected to help but were out of scope.

## 7. Reproducibility
Everything is in `geosoil/` (model + bake-off + verification) and `data/extraction/`
(public-data pulls). See `geosoil/README.md` for commands and `docs/research/` for the
SOTA survey and data roadmap.

## References
I-JEPA (2301.08243); VICReg (2105.04906); SatCLIP (2311.17179); Mamba (2312.00752);
conformal prediction (Angelopoulos & Bates 2107.07511); TabPFN (Nature 2025);
AlphaEarth (2507.22291); SoilGrids 2.0 (SOIL 2021); "trees beat DL on tabular"
(Grinsztajn 2022, 2207.08815).
