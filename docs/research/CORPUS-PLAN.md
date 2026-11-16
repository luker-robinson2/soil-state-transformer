# Building the GeoSoil corpus: sources, sampling, features

Draft 2026-11-16. This is the data half of SCALING-PLAN.md. Corpus tiers C0–C3 and rungs R0–R4 are
defined there. Every source used must be recorded in `data/corpus/sources.yaml` with its license
before any of its data enters the corpus.

Verification status: counts and licenses were checked against the source pages on 2026-11-16 unless
marked *(unverified)*. Re-check before a release.

## A. Labelled soil data (the probe and fine-tune set)

| Priority | Source | Size | Targets | License / access | Notes |
|---|---|---|---|---|---|
| 1 | **NCSS/KSSL Lab Data Mart** | ~65k profiles (soilDB docs); layers *(unverified, ~400k)* | all 5, at depth, plus MIR | Public domain. SQLite or GeoPackage from ncsslabdatamart.sc.egov.usda.gov, monthly | We use ~3k today. Biggest single win. |
| 2 | **WoSIS 2023 snapshot** | 228k profiles, 217k sites, 174 countries, >900k layers | all 5 | **Per-profile CC-BY or CC-BY-NC.** Filter on the license column. WFS or DOI 10.17027/isric-wdcsoils-20231130 | We have the CONUS slice (37k) on disk, unused. Its largest contributor is US NCSS, so it overlaps KSSL. |
| 3 | **OSSL v1.2** | 135,651 samples, 87,707 with coordinates. MIR 82,573, VisNIR 64,644 | SOC, pH, texture, BD + spectra | Mixed CC-BY / CC-BY-SA / ODbL. **The LUCAS rows inherit the JRC license.** | The KSSL slice duplicates LDM; join on `labsampnum`. Spectra become an optional proximal modality. |
| 4 | **RaCA** | 6,017 sites, 32,084 pedons, 144,833 samples, to 100 cm | SOC (+ BD, VisNIR) | Public domain, 336 MB | **Site coordinates are degraded on purpose.** Use for SOC-stock and depth supervision and as the final hold-out, not for 10 m pixel matching. |
| 5 | **LUCAS topsoil** 2009/2015/2018/2022 | ~20k / 22k / 19k *(unverified)* / 33k locations | all 5 (2022: + metals, eDNA) | Free, but **request form + license; no redistribution** | The KSSL → LUCAS transfer test. Never commit LUCAS rows. |
| 6 | **USGS DS 801** (conterminous US soils) | 4,857 sites, 3 depths | geochemistry, mineralogy | Public domain | Small, but a probability-like national sample. Good for design-based accuracy checks. |
| 7 | AfSIS / iSDA | ~1.9k with spectra + more *(unverified)* | | CC-BY 4.0, `s3://afsis` | Africa coverage for the global tier. |
| 8 | Brazil febr (14,477), BDsolos (~9k); Canada NPDB; Australia ANSIS / TERN | various | | Mostly open (Canada OGL; TERN CC-BY) | Global tier. ANSIS federates the Australian state collections. Check its API. |
| 9 | NEON megapit + distributed soils | 47 sites *(unverified)* | | CC0 | Research grade, co-located with flux towers. |

**Pseudo-labels: pretraining ablation only, never evaluation.** gNATSGO/SSURGO (public domain),
POLARIS (p5/p50/p95), SoilGrids 2.0 (CC-BY) and OpenLandMap (**CC-BY-SA**). These are model products.
Training on them is how the current model learned to copy OpenLandMap (lab SOC R² −0.44).

**Deduplication is mandatory.** These sources overlap:
- OSSL ⊃ KSSL, LUCAS
- WoSIS ⊃ NCSS
- ISCN ≈ re-packaged NRCS
- SoilGrids training points = WoSIS
- RaCA samples were run at KSSL

Order of keys:
1. Source keys (`pedon_key`, `labsampnum`).
2. Coordinates within 100 m **and** an overlapping depth interval.
3. Keep the record with the most complete method metadata.

Log every drop in `data/corpus/dedup_log.parquet`.

After dedup, expect about **250k unique labelled locations** globally, about 80k of them in CONUS *(estimate)*.

## B. Inputs: gridded sources and open foundation models

**Read from the cloud, not Earth Engine.** That keeps the EE quota for AlphaEarth fallbacks only.

| Modality | Source | Access | Cadence used |
|---|---|---|---|
| Climate | **ERA5-Land** (ARCO Zarr via Earth Data Hub, free token; or CDS), ARCO-ERA5 `gs://gcp-public-data-arco-era5` (anonymous), Daymet / gridMET / TerraClimate (Planetary Computer), PRISM (wget) | Zarr point reads | daily → weekly patches |
| Vegetation, LST | MODIS 6.1 NDVI/EVI/LST (Planetary Computer STAC) | COG | 8/16-day → weekly |
| Soil moisture | SMAP L3/L4 | NASA Earthdata `earthaccess` | daily → weekly |
| Terrain | Copernicus DEM 30 m, USGS 3DEP (Planetary Computer / AWS) | COG | static: elevation, slope, TWI, curvature, MRVBF |
| Soil-sensing | **USGS aeroradiometric K/U/Th** grids, bare-soil S2 composite (SYSI/BSC method), S1 RTC SAR (Planetary Computer), surficial geology | various | static / annual |
| Land cover | ESA WorldCover, Dynamic World (EE only), CDL (CONUS) | COG / EE | annual |
| **AlphaEarth** embeddings | 64-d, 10 m, 2017–2025, CC-BY 4.0. **Source Cooperative mirror `source.coop/tge-labs/aef`** (free). `gs://alphaearth_foundations` has been requester-pays since 2026-07. | COG range reads, `aef-loader` | annual |
| **TESSERA** embeddings | 128-d Matryoshka, 10 m, S1+S2, 2017–2025, CC-BY (weights CC0, plus an Acceptable Use Policy) | `pip install geotessera` | annual |

**Open pretrained models to use:**

| Model | Why | How it enters GeoSoil |
|---|---|---|
| **Galileo** (nasaharvest, Nano / Tiny / Base 85M) | Pixel time series over S1, S2, ERA5, TerraClimate, SRTM and Dynamic World: the closest match to our inputs. Pretraining data released. | Baseline, and initialization for the temporal encoder (R1 ablation) |
| **TESSERA** (1M–44M students, CC0) | Pixel S1/S2 series, distillation recipe | Second embedding modality, baseline, and the distillation recipe for R4 |
| **Presto** (~0.4M) | Tiny monthly pixel time-series model | Fast baseline |
| AnySat, Prithvi-EO-2.0, TerraMind, Copernicus-FM | Patch / image models | Low priority; GeoSoil is per-point |

Licenses marked "verify" in the survey (Galileo, Presto, AnySat, Copernicus-FM) must be confirmed from
each repo before use. Aurora and Prithvi-WxC are weather models and not relevant.

**Spatial or patch embeddings, not needed now:** Clay v1.5 (chip-level), Major TOM embedding sets.

## C. Sampling the unlabelled corpus (C2 and C3)

The goal is to cover as many distinct soil-forming conditions as possible per byte, not to sample uniformly.

1. **Strata.** Use climate (Köppen or k-means on ERA5 normals), terrain class, land cover, and an
   AlphaEarth k-means cluster (k ≈ 256).
2. **Draw** with conditioned Latin hypercube sampling (cLHS; Minasny & McBratney 2006) or iterative
   proportional fitting over the strata (as in Wickrema et al. 2025). The aim is to widen the range of
   variation rather than duplicate common regimes.
3. **Thin near-duplicates** by AlphaEarth embedding distance, SemDeDup-style (arXiv 2303.09540). Sorscher
   et al. 2022 show pruning helps most when data is plentiful, so thin C3 hard and C2 lightly.
4. **Always include** every labelled site, plus 4–8 unlabelled neighbours within 1 km. Pretraining then
   covers the label distribution, and the neighbours provide local-contrast pairs.
5. **Hold out** 1° blocks (about 10%) and the year 2025 across all tiers. Fix the block assignment once,
   in `data/corpus/splits.parquet`.

Sizes: C2 = 2M CONUS locations, C3 = 10M global. Go beyond that only if the R2 → R3 probe curve is still
rising with D at fixed N.

## D. Choosing features

Start from SCORPAN (soil = f(soil, climate, organisms, relief, parent material, age, space)) and treat
every candidate as a **group** (a modality), not as a single column.

| SCORPAN factor | Candidate group | Current status |
|---|---|---|
| s (soil) | OSSL spectra (where present), pseudo-label priors (ablation only) | not built |
| c (climate) | ERA5-Land weekly series, climate normals, aridity index | 12-month version built |
| o (organisms) | MODIS phenology, AlphaEarth, TESSERA, land cover / CDL history | AlphaEarth + CDL built |
| r (relief) | elevation, slope, TWI, curvature, MRVBF | partly built |
| p (parent material) | **gamma radiometrics**, surficial geology, bare-soil S2, S1 SAR | bare-soil + SAR built |
| a (age) | geologic age (from the geology map) | not built |
| n (space) | Fourier lon/lat features | built |

**Selection protocol.** The evidence favours a hybrid: about 10–20 physical covariates plus the
embeddings, rather than either alone (Koch et al. EGU26-13089; SpTGNN arXiv 2606.16580).

1. **Groupwise permutation importance and grouped SHAP, computed inside 5-fold spatial-block CV**, for
   LightGBM on lab labels.
   - Importance from random CV rewards features that just proxy location (Ploton et al. 2020).
   - Report importance per target. pH and texture lean on parent material and climate; SOC leans on
     climate, vegetation and relief.
2. **Drop-group ablation** for the neural model, which modality dropout during pretraining gives
   almost for free.
3. **Keep a group** if it improves lab-label spatial-CV skill for at least one target by more than the
   seed-to-seed noise, or widens the area of applicability.
   Drop a group if it only helps on OpenLandMap labels.
4. **Temporal granularity test (R1):**
   - (a) climate normals plus summary statistics
   - (b) monthly series
   - (c) weekly patches over 1–4 years
   Expect (a) ≈ (c) for texture and pH. Keep the long series only for targets where it beats (a).
5. **Validation reporting:** always report spatial-block CV, leave-region-out, and area-of-applicability
   masks. Where a probability sample exists (RaCA, USDA DS 801), add design-based accuracy
   (Wadoux et al. 2021).

## E. Label harmonization

- **Keep the method, don't overwrite the value.** Carry `method_id` for every label. WoSIS codes the
  analytical method for every value; KSSL has lab procedure codes.
- **SOC:**
  - Walkley-Black vs dry combustion: the correction is not a flat 1.32. It ranges 0.9–1.5 and rises with
    clay (Shamrikova / FAO GSP; J. Plant Nutr. Soil Sci. 2025).
  - Prefer a learned per-method offset head over converting values up front.
- **pH:** H₂O vs CaCl₂. Use an EC-aware conversion (Minasny et al. 2011), or a separate head per method.
- **Depth:** fit equal-area splines to standard depths (0–5, 5–15, 15–30, 30–60, 60–100 cm). The latent
  already takes depth as an input.
- **Noise floor:** estimate it from KSSL lab replicates per target, and store it in the results.

## F. Finding more data (outreach and lesser-known sources)

These are leads to check; none has been verified yet. Record each in `sources.yaml` with
`status: lead` until checked.

- **National and federal:**
  - NRCS soil survey data requests (unpublished pedons, precise RaCA coordinates under a data-use agreement)
  - USDA-ARS LTAR network
  - Soil Health Institute (North American Project to Evaluate Soil Health Measurements)
  - USGS National Geochemical Database
  - EPA NARS wetland soils
- **Research networks:**
  - CZNet (Critical Zone Collaborative Network)
  - LTER sites
  - AmeriFlux / FLUXNET site soil data
  - ISCN Gen3
  - NEON
- **Data papers:** search Zenodo, PANGAEA, ESSD and Scientific Data for "soil profile database" or
  "soil organic carbon dataset". Each country often has one (China, India, Europe beyond LUCAS, Russia's
  EGRPR, New Zealand's NSD).
- **Through ISRIC:** ask about WoSIS profiles with restricted licenses. Some holders grant academic use on request.
- **CU Boulder:** Greg Tucker's group and INSTAAR may hold Front Range or Critical Zone soil cores; CSDMS
  lists community datasets.
- **Spectral libraries:** additional MIR/VisNIR sets submitted to OSSL after v1.2, and the KSSL MIR
  library models (arXiv 2511.15965).

## G. Work items (for contributors)

Each item lands as one PR with a test or a reproducible notebook. The source must be in `sources.yaml`
before any code that downloads it.

| ID | Item | Output | Runs on |
|---|---|---|---|
| C0.1 | Copy the GEE parquets to Alpine; rebuild master.parquet with temporal modalities | master.parquet | laptop → dtn |
| C0.2 | Measure bytes and time per point for AlphaEarth range reads from Source Cooperative (`aef-loader`), versus EE `sampleRegions` | a short measurement note | acpu |
| C0.3 | Test outbound internet from compute nodes (printed by the bootstrap job) | yes/no | aa100 |
| C1.1 | KSSL LDM ingest (SQLite) → harmonized label table with `method_id` and depth splines | `labels_kssl.parquet` | acpu |
| C1.2 | WoSIS 2023 ingest, CC-BY filter, dedup against KSSL. Start from `foundation-model/scripts/prepare_wosis.py`, which already loads global WoSIS for SST. | `labels_wosis.parquet` + dedup log | acpu |
| C1.3 | OSSL ingest + spectra modality (optional) | `labels_ossl.parquet`, `spectra.zarr` | acpu |
| C1.4 | RaCA ingest as SOC-stock hold-out | `labels_raca.parquet` | acpu |
| C1.5 | LUCAS (after the license request) as a transfer test; never committed | local only | acpu |
| C1.6 | Point feature extractors: ERA5-Land, MODIS, SMAP, DEM, radiometrics, AlphaEarth, TESSERA → feature store | `features/` Zarr shards | acpu array |
| C1.7 | Spatial-block + 2025 splits | `splits.parquet` | acpu |
| C1.8 | Feature-selection report (§D, steps 1–4) | `results/feature_selection.md` | acpu |
| C2.1 | Strata + cLHS sampler + embedding-distance thinning | `points_c2.parquet` | acpu |
| C2.2 | Run the C1.6 extractors at 2M points | feature store | acpu array |
| C3.x | Same as C2 for global, 10M points | feature store | acpu array, after Ascent |
