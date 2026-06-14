# SOTA survey — multi-modal geospatial soil ML

Consolidated synthesis from a multi-agent literature review (June 2026), scoped to
our setting: small-n (~3k), feature-vector modalities (AlphaEarth 64-d + Sentinel-2
+ covariates), multi-target soil prediction, spatial-block CV, verified against lab
truth. Only well-established references are cited; speculative architectures are
described by mechanism.

## 1. Architectures

| Idea | Verdict for our setting |
|---|---|
| **JEPA** — predict in *latent* space (vs reconstructing inputs); EMA target encoder. I-JEPA (arXiv 2301.08243). Tabular variants extend this to feature subsets. | **Adopt as the core.** Cross-modal "predict one modality's embedding from another" is a natural, augmentation-free objective for AlphaEarth↔Sentinel-2. |
| **Contrastive (CLIP/InfoNCE)**; geospatial: SatCLIP (arXiv 2311.17179). | **Adopt** as cross-modal alignment loss. |
| **VICReg** variance/covariance regularization (arXiv 2105.04906); Barlow Twins. | **Adopt** to prevent latent collapse at small batch size. |
| **Mamba / selective SSMs** (arXiv 2312.00752). | **Only for temporal sequences** — not static vectors. Applicable once ERA5/MODIS monthly series are added; compared as a temporal-encoder swap. |
| **Energy-based models** (NCE; Gutmann & Hyvärinen 2010). | **Research branch** — conditional energy head for calibrated UQ + OOD scoring; compared on the same latent. |

## 2. Tabular baselines (the bar to beat)

- **Gradient-boosted trees still lead on tabular** (Grinsztajn et al., NeurIPS 2022, arXiv 2207.08815): robust to uninformative features, handle non-smooth targets. LightGBM / XGBoost / CatBoost are the baselines to beat.
- **TabPFN-v2** (Hollmann et al., *Nature* 2025) is SOTA for *small-n* tabular and specifically strong for **field-scale digital soil mapping** (arXiv 2508.09888). (Gated HuggingFace model.)
- **Hybrid**: feed a neural encoder's frozen latent into a GBM — often the strongest combination on small-n.
- **Spatial-block CV is mandatory**: random CV inflates soil metrics 10–25% via spatial autocorrelation.

## 3. Metric targets (spatial CV, AlphaEarth+S2-grade inputs)

| Property | "Good" R² band | Notes |
|---|---|---|
| SOC | 0.75–0.85 | log-transform; right-skewed |
| pH | 0.80–0.88 | best-predicted |
| clay | 0.72–0.82 | |
| sand | 0.68–0.78 | |
| bulk density | 0.65–0.75 | sparsest labels |

Also report **RPIQ** (IQR/RMSE): >2.0 good, >2.5 excellent.

## 4. Geospatial foundation embeddings

- **AlphaEarth Foundations** (Google DeepMind; arXiv 2507.22291): 64-d, 10 m, annual; CC-BY. Best used as **frozen features**. We already use it.
- Others to combine later (need raw imagery + more compute): **Prithvi-EO-2.0** (IBM/NASA), **Clay v1.5**, **SatMAE/++**, **Presto**, **Galileo**. Freeze-and-probe is the right pattern at small n.

## 5. Ground truth to cross-reference (multiple independent truths)

| Source | What | Access | Role here |
|---|---|---|---|
| **OpenLandMap** | ML-derived SOC/pH/texture/BD | on disk | training labels (a *model product* — verify against lab) |
| **KSSL/NCSS** | lab-measured pedons (US) | on disk + ncsslabdatamart | **primary lab truth** |
| **WoSIS** (ISRIC) | harmonized global profiles | on disk | global cross-check |
| **RaCA** (USDA) | designed carbon survey (~6k US) | Ag Data Commons | independent SOC truth |
| **OSSL** | >100k spectral lab samples | soilspecdata | independent lab truth |
| **Covariates** | ERA5 climate, Copernicus DEM, MODIS NDVI | Earth Engine | temporal + terrain modalities |

Harmonize to **0–30 cm** (equal-area spline), normalize units, join on (lon, lat).

## Key takeaway

On *static* tabular vectors, GBMs are hard to beat — so the neural model earns its
keep through (a) the **learned multi-modal representation** (probes/retrieval/
transfer), (b) **cross-modal self-supervision** (JEPA + InfoNCE), (c) **temporal
sequence modeling** (Mamba) that GBMs handle poorly, and (d) **calibrated
uncertainty**. The honest scientific story is the cross-source generalization gap
(OpenLandMap labels vs lab truth), which only multi-truth validation reveals.
