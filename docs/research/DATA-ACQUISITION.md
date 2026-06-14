# Data acquisition plan — more training data + CU Boulder access

Deep multi-source hunt (June 2026) for additional training/validation data for
**GeoSoil** (multi-modal soil model) and the **Markov food-insecurity** project,
with emphasis on what a **CU Boulder** affiliation unlocks. Ranked by value × ease.
Confidence flags from the source agents; verify license/coords at download.

---

## ★ The biggest unlock: free A100 GPUs (CU Research Computing — Alpine)

The single highest-value find. Directly fixes the "Mamba/large models too slow on a
Mac CPU" bottleneck, **free**, and **no PI/sponsor needed**.
- **33× NVIDIA A100** (partition `aa100`) + AMD MI100s; up to 15 GPUs/job, 7-day walltime.
- New users auto-get a **~2,000 SU/month Trailhead allocation** (no application).
- **Get it this week:** request account at
  `https://rcamp.rc.colorado.edu/accounts/account-request/create/organization`
  → Duo 2FA → `ssh <identikey>@login.rc.colorado.edu` (or Open OnDemand JupyterLab)
  → Slurm `--partition=aa100 --gres=gpu:1`. Docs: `https://curc.readthedocs.io`.
- **Blanca** condo (ask advisor) may give newer H200/RTX-6000 priority. `/scratch/alpine`
  storage is free; PetaLibrary for large persistent datasets.

---

## A. Soil ground truth — independent, LAB-measured (the core need)

The central GeoSoil finding is weak lab-verifiable signal for SOC/texture, so more
independent lab pedons are the top lever. **Deduplicate aggressively** (OSSL ⊃ KSSL/
LUCAS/AfSIS; WoSIS ⊃ AfSP; NCSS ≈ KSSL) — track provenance UUIDs to avoid CV leakage.

| Get first | What | Access |
|---|---|---|
| **1. OSSL** | >110k georef lab samples (SOC/pH/texture/BD/CEC/N) + VNIR/MIR spectra; MIT/CC-BY | `wget https://storage.googleapis.com/soilspec4gg-public/ossl_all_L0_v1.2.csv.gz` (static GCS files; avoid the experimental API). Dedupe on `id.layer_uuid_txt`. |
| **2. NEON soil** | CONUS, **KSSL-analyzed** (same lab as our truth), open, full GPS, repeat sampling; **Boulder HQ** | R/Py `neonUtilities`/`neonutilities` → `DP1.10047.001` (initial), `DP1.10086.001` (periodic), `DP1.10096.001` (megapit) |
| **3. RaCA** | 6,017 CONUS sites, **bulk density** + SOC; **public coords degraded** | `soilDB::fetchRaCA()` for central-pedon coords; pursue a direct NRCS/KSSL full-precision request (CU/USDA contact helps) — **biggest single CONUS lever if coords recovered** |
| **4. LUCAS Soil (EU)** | ~19–22k harmonized points, repeated 2009/15/18/22; pretraining gold | ESDAC registration: `https://esdac.jrc.ec.europa.eu/projects/lucas` |
| **5. AfSIS** | ~18k Sub-Saharan georef + spectra; tropical diversity | `aws s3 ls --no-sign-request s3://afsis/` |
| also | **WoSIS global** snapshot (we only use CONUS) `wosis_202312.gpkg`; **ISCN v3** (extra BD); USGS geochem (DS 801, auxiliary) | ISRIC DOI / iscn.fluxdata.org |

---

## B. Soil-sensing / proximal modalities (recover the texture/SOC signal)

These physically sense the *profile*, not the surface — the documented frontier.

| Get next | What it senses | Access |
|---|---|---|
| **1. NURE gamma radiometrics (K/eU/eTh + uncertainty)** | **parent material & texture** (Th↔clay; gamma+RF reaches R²≈0.87 for clay) — the one modality filling the real gap; CC0 | USGS ScienceBase Bayesian grid (DOI 10.5066/P9YEAFHI) or `mrdata.usgs.gov/radiometric/`; **not in GEE** — download GeoTIFF, reproject, sample. Coarse (~1–2.5 km) parent-material covariate. |
| **2. POLARIS 30 m** | texture/SOM/BD/pH priors, 6 depths; CONUS | GEE `projects/sat-io/open-datasets/polaris/<prop>_mean` (om/ksat in log10; mirror tagged CC-BY-NC) |
| **3. gNATSGO 10 m + `mukey`→SSURGO horizon join** | authoritative *measured* texture/SOC by depth; CC0 | GEE `projects/sat-io/open-datasets/gNATSGO/raster/*`; join mukey for horizon tables |
| **4. EMIT / AVIRIS hyperspectral** | mineral/organic absorption | GEE `NASA/EMIT/L2A/RFL`; L2B mineralogy via `earthaccess` (LP DAAC). **Sparse swaths** → per-point "feature where available", not a national layer |
| **5. SoilGrids 250 m + uncertainty; 3DEP 10 m terrain; gridMET 4 km / Daymet 1 km** | independent priors; finer terrain & climate | all in GEE (`ISRIC/SoilGrids250m/v2_0`, `USGS/3DEP/10m`, `IDAHO_EPSCOR/GRIDMET`, `NASA/ORNL/DAYMET_V4`); TAGEE for terrain derivatives |

PRISM 800 m is paid — **ask CU Libraries (Phil White) if CU holds a subscription**;
else use Daymet/gridMET free. Planet 3 m via the **Education & Research program**
(`.edu` qualifies; ~3,000 km²/mo) for targeted bare-soil snapshots, not a CONUS layer.

---

## C. Food-insecurity / famine data (the Markov project)

| Get first | What | Access |
|---|---|---|
| **1. HFID** | modeling-ready monthly panel: IPC/CH + FEWS NET phases + WFP FCS/rCSI on GADM admin-2, 2007–present, 80 countries, 311,838 rows + geometries — **your Markov states out of the box** | Zenodo DOI 10.5281/zenodo.15017473 (also HDX) |
| **2. FEWS NET FDW API** | longest IPC-compatible phase history (2009+) | `https://fdw.fews.net/api/` (JSON/CSV) |
| **3. IPC/CH API** | official consensus phases + Cadre Harmonisé | key at `ipcinfo.org/ipc-country-analysis/api/` (slow approval — request now) |
| covariates | CHIRPS rainfall, MODIS NDVI, NOAA VHI, SPEI drought, **ACLED** conflict (free academic w/ `.edu`), WFP/FAO prices, IOM-DTM/UNHCR displacement, USDA ERS US food security | mostly GEE + APIs (zonal-average onto HFID polygons in GEE) |

Reference architectures: HFID paper (Nature Sci Data 2025), Westerveld 2021
(livelihood-zone XGBoost), CERES (0.25° CHIRPS+NDVI+ACLED). Caveat: FEWS NET phases
are IPC-*compatible*, not IPC-consensus — keep as separate state sources.

---

## D. CU Boulder access — act-this-week checklist

1. **Alpine HPC account** (above) → free A100s, no sponsor. ★ top priority.
2. **Google Earth Engine** `.edu` register + complete noncommercial verification
   (projects unverified after 2025-09-26 may be on hold).
3. **NEON** soil+flux (`neonutilities`) — Boulder HQ; email re: workshops/internships.
4. **NASA Earthdata login → SMAP** via `earthaccess`/**NSIDC** (on CU campus/CIRES).
5. **NCAR RDA/GDEX** account → ERA5/reanalysis via THREDDS streaming.
6. **CU Libraries**: free ArcGIS (AGOL + Pro); email GIS librarian **Phil White** re:
   PRISM/Planet/restricted licenses; Statista/Sage Data/NHGIS for socioeconomic layers.
7. **Microsoft Planetary Computer** = **data/STAC API only** (the free compute Hub
   retired 2024) — run compute on Alpine.
8. **USDA-ARS Fort Collins** (~1 hr) — labeled soil data via a CU soil-faculty intro.
9. **Skip** the $11.6k Earth Data Analytics certificate — materials free at
   `earthdatascience.org`.

---

## Recommended order of operations (highest ROI first)
1. **Alpine account** (compute) + **GEE verify** — unblocks everything.
2. **OSSL `wget`** + **NEON pull** — biggest, easiest lab-truth gains today.
3. **NURE radiometrics + POLARIS + gNATSGO** — the soil-sensing modalities that target
   the texture/SOC gap (radiometrics is the needle-mover).
4. **RaCA full-precision request** (slow; start now) — 6k independent CONUS BD sites.
5. **HFID + GEE covariate reducer** — stands the Markov project up immediately.
6. LUCAS/AfSIS/WoSIS-global for pretraining breadth.
