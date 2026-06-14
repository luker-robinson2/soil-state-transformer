# Data roadmap — toward more verifiable signal

Motivated by the central finding: a model can score R²≈0.75 SOC / 0.79 sand against
**OpenLandMap** labels yet collapse to ≤0 against **lab-measured** truth, because
AlphaEarth encodes land-*surface* appearance, not subsurface soil. Only **pH**
showed genuine lab-verifiable signal. Two levers fix this:

1. **More independent lab ground truth** — train/validate on *measurements*, not model products.
2. **Modalities that physically sense the soil**, not just its surface canopy.

Sources are ranked by *verifiable-signal-per-effort* for our CONUS / single-Mac setup.

## A. Independent ground truth (more truths to cross-reference)

| # | Source | Adds | Access | Effort |
|---|---|---|---|---|
| 1 | **RaCA** (USDA Rapid Carbon Assessment, ~6k US sites, designed sample) | independent **SOC + BD** truth, statistically unbiased | Ag Data Commons CSV / `soilDB::fetchRaCA()` | low |
| 2 | **Full KSSL/NCSS** (~60k US pedons; we use ~3k) | far more lab SOC/pH/texture/BD + **MIR spectra**, depth horizons | NCSS Lab Data Mart SQLite / `soilDB::fetchLDM()` | low–med |
| 3 | **OSSL** (>100k global lab samples + VNIR/MIR spectra) | huge lab truth **and** the spectral modality (below) | `soilspecdata` (Python) | med |
| 4 | **LUCAS Soil** (EU, ~45k) | **cross-continent transfer test** (train US → test EU) — the strongest generalization check | ESDAC download | med |
| 5 | **ISRIC WoSIS** (global profiles; we have CONUS) | global validation, soil-order stratification | WoSIS API / on disk | low |
| 6 | **NEON megapit + periodic soils** | research-grade, co-located with flux/sensor towers | NEON data portal | med |

**Verification upgrades these enable:** leave-whole-region-out and leave-soil-order-out
transfer; train-US/test-EU; a measurement **noise floor** from KSSL lab replicates
(don't claim R² above what duplicate lab samples agree to).

## B. Modalities that sense the soil itself (the missing signal)

| # | Modality | Why it carries *soil* signal AlphaEarth lacks | Access |
|---|---|---|---|
| 1 | **Gamma-ray radiometrics** (airborne K/U/Th) | directly senses surface **mineralogy / parent material / texture** — the texture signal that's invisible optically | USGS national aeroradiometric grids (reproject); Australia's is in GEE as a template |
| 2 | **Soil spectroscopy (MIR/VNIR)** | lab-gold standard: MIR → SOC/texture/pH at R²>0.9 | OSSL / KSSL (lab-measured, not remote — a "proximal" branch) |
| 3 | **Bare-soil composites** (multi-year Sentinel-2, vegetation-masked) | actual **soil reflectance** vs canopy; SOC/clay correlated | GEE (extend our S2 extractor: NDVI<0.25 ∧ NBR2<0.075 median) |
| 4 | **Sentinel-1 SAR** (VV/VH, multi-temporal) | surface **roughness / moisture / texture** sensitivity, all-weather | GEE `COPERNICUS/S1_GRD` |
| 5 | **Imaging spectroscopy from space** (EMIT, EnMAP, PRISMA) | hyperspectral **soil mineral** absorption features | EMIT (GEE/LP DAAC), EnMAP/PRISMA scenes |
| 6 | **Parent material / surficial geology, soil-order maps** | the lithology prior texture/BD depend on | SoilGrids covariates, USGS geology, gNATSGO/POLARIS |
| 7 | **Other geospatial FM embeddings** (Prithvi, Clay, Presto) | complementary learned features; ensemble of embeddings | HuggingFace (need raw imagery + more compute) |

## C. Depth & dynamics

- **Multi-depth labels**: KSSL/RaCA have horizons — predict 0–5/5–15/15–30 cm with an
  equal-area spline target, not just topsoil. The latent already takes depth as input.
- **Multi-year temporal**: extend ERA5/MODIS/S1 to multi-year sequences → the Mamba
  branch models *dynamics* (carbon change), which is where sequence models earn their keep.

## Recommended next 3 steps (highest ROI)

1. **RaCA + full-KSSL ingestion** → AlphaEarth + new covariates extracted at their points
   → adds two independent lab truths and ~10× more lab labels. Lets us *train on lab
   truth directly* and report honest in-domain lab R².
2. **Gamma radiometrics + bare-soil Sentinel-2 + Sentinel-1** → the soil-sensing modalities
   most likely to recover the texture/BD signal AlphaEarth misses. This is the single most
   promising signal upgrade.
3. **OSSL spectral branch** → a small spectral encoder pretrained on MIR, fused as a modality;
   where spectra exist this should lift every property toward its lab ceiling, and gives a
   third independent truth.

Each plugs into the existing pipeline: extract at points → `geosoil/config.py` modality
columns → `data.py` auto-joins → bake-off re-runs. No architecture change required.
