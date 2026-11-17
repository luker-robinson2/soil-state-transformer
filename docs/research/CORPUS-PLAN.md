# Building the GeoSoil corpus: sources, acquisition, sampling, features

Draft 2026-11-16, expanded the same day. This is the data half of SCALING-PLAN.md, which defines corpus
tiers C0–C4 and rungs R0–R4 and maps each tier to a model size. The step-by-step schedule is
EXECUTION-PLAN.md.

Rules:
- Every source used must be registered in `data/corpus/sources.yaml` with its license before any of its
  data enters the corpus.
- Status markers: **[V]** = verified on a primary or near-primary page on 2026-11-16; **[U]** = not
  verified. Re-check before a release.

## A. Labelled soil data: every large source we know of

Ranked by how soon we can actually use it, not by raw size. The two largest datasets in the world are
not open (India SHC, China's 3rd survey); see §A3.

### A1. Open now: download and ingest

**Status 2026-11-16:** C1 is built (`python -m geosoil.corpus.build_c1`).
- **Ingested:** KSSL (labtaxa mirror), WoSIS 2023, Canada NPDB, Brazil SoilData 2024, OSSL (non-KSSL, non-LUCAS rows),
  NAPESHM, USGS DS 801 and RaCA (hold-out). Each was pulled anonymously.
- **Arriving via WoSIS:** Australia (CSIRO), Mexico (INEGI), AfSP and Chile.
- **Not pullable:** ISCN3, LUCAS-MEGA, CSDLv2 profiles, INEGI bulk, TERN, NEON and NZ NSD. These need a login, aren't
  released, or the link is dead. `data/corpus/sources.yaml` records why for each.

| Source | Size | Region | Targets | License | Notes |
|---|---|---|---|---|---|
| **NCSS/KSSL Lab Data Mart** | ~65k profiles; layers ~400k [U] | US + some foreign | all 5 at depth, + MIR | public domain | SQLite / GeoPackage, monthly. We use ~3k today. **Biggest single US win.** |
| **WoSIS 2023** | 228k profiles, 217k sites, 174 countries, >900k layers [V] | global | all 5 | per profile: CC-BY or CC-BY-NC | WFS or DOI 10.17027/isric-wdcsoils-20231130. The CONUS slice (37k) is on disk, unused. Overlaps NCSS. |
| **OSSL v1.2** | 135,651 samples, 87,707 georeferenced; MIR 82–92k, VisNIR 64–65k [V] | global | SOC, pH, texture, BD + spectra | CC-BY / CC-BY-SA / ODbL. LUCAS rows: JRC license. | Contains KSSL (join on `labsampnum`) and LUCAS. |
| **ISCN3** | 71,198 profiles, 431,324 layers, 67,112 sites [V] | mostly US | SOC, BD | per contributor | EDI `edi.360`. Heavily overlaps NRCS, so dedupe against KSSL. |
| **Brazil SoilData / MapBiomas Solo** | 45k+ samples at 15k points; 28,065 SOC-stock points [V] | Brazil | texture, OC, BD, coarse fragments | mostly CC-BY | soildata.mapbiomas.org. **Best new open win:** tropical soils we lack entirely. |
| **iSDA / AfSIS** (open part) | ~49k entries open [U]; ~100–150k trained iSDAsoil [V] | sub-Saharan Africa | chemistry + MIR | CC-BY | AWS `s3://isdasoil`, `s3://afsis`, SoilHive. The full set is on request (§A2). |
| **RaCA** | 6,017 sites, 32,084 pedons, 144,833 samples to 100 cm [V] | US | SOC, BD, VisNIR | public domain | **Coordinates degraded on purpose.** Use as the final SOC hold-out and for depth/stock supervision. |
| **LUCAS-MEGA** | 72,552 samples at 29,872 points [V] | EU | harmonized LUCAS 2009–2018 | open (arXiv 2512.09576) | Check whether its license overrides the JRC terms before committing any row. |
| **LUCAS 2022 topsoil** | 18,984 samples [V] | EU | all 5 + metals, eDNA | ESDAC registration, **no redistribution** | The ESDAC page says public release is likely Q1 2027 [V]. KSSL → EU transfer test. Never commit rows. |
| **Mexico INEGI** | 62,362 SOC values at 26,297 sites [V] | Mexico | SOC (+ others) | INEGI open data | Partly inside WoSIS already. |
| **China CSDLv2** | 11,209 profiles [V] | China | all 5 | open (scidb.cn) | May release only gridded layers [U]. Overlaps WoSIS. |
| **Canada NPDB** (open subset) | count not stated; >10k pedons behind the 100 m grids [V] | Canada | | Open Government Licence | open.canada.ca. Full holdings on request. |
| **Australia TERN Soil Data Federator** | 195,383 site observations [V]; lab-analysed subset [U] | Australia | varies | mostly CC-BY | Free API key. ANSIS is the newer portal. |
| **USGS DS 801** | 4,857 sites, 3 depths | CONUS | geochemistry, mineralogy | public domain | A probability-like national sample, used for design-based accuracy. |
| **Soil Health Institute NAPESHM** | 1,453 plots, 94 sites, 110 variables [V] | US/Mexico | soil health + management | open (Ag Data Commons) | Small, but rich in management treatments. |
| New Zealand NSD | ~1,500 profiles [V] | NZ | | free | |
| NEON | 47 sites [U] | US | | CC0 | Co-located with flux towers. |

**Spectral libraries** (an optional proximal modality and extra SOC/texture truth): KSSL MIR (>80k,
inside OSSL), Brazil BSSL (39,284 [V], on request), LUCAS VisNIR (~20k per round), AfSIS MIR,
Viscarra Rossel global VisNIR (~23k [U], on request), ICRAF-ISRIC (~4.4k [U]).

### A2. On request: weeks to months; send these first, they are the long pole

| Source | What you get | How |
|---|---|---|
| iSDA full sample table | ~100–150k African samples (vs ~49k open) | Email iSDA; noncommercial data agreement |
| Brazil BSSL | 39k spectra + chemistry | Email J. Demattê (University of São Paulo, ESALQ); offer co-authorship |
| WoSIS restricted profiles | profiles whose providers didn't grant open release | Email ISRIC (WoSIS team); they broker provider consent |
| Canada NPDB full | full pedon holdings | aafc.cansis-siscan.aac@agr.gc.ca |
| Viscarra Rossel global VisNIR | ~23k global spectra | Email the authors |
| RaCA exact coordinates | true site locations | NRCS National Soil Survey Center, as a National Cooperative Soil Survey (NCSS) cooperator; no published procedure [U] |
| USFS FIA Phase-3 soils, true plot coordinates | forest pH / C / N plots; fills KSSL's forest gap | FIA Spatial Data Services. Needs a study plan + an FIA collaborator. Cheaper route: SDS runs our covariate overlay for us. |
| EnMAP scenes | hyperspectral soil minerals | DLR noncommercial science proposal (archive access is immediate) |

### A3. Largest in the world, but not open: collaboration only

| Source | Size | Status |
|---|---|---|
| **India Soil Health Card** | ~50M samples [V]; GPS since 2023 | AIKosh lists it "Restricted"; public portal is aggregates only. Route: AIKosh access (needs an Indian institution) or a collaboration with ICAR-IISS Bhopal / IIT Delhi (CoRE Stack). No texture/BD; mixed methods. **Do not scrape.** |
| **China 3rd National Soil Survey** | 2.87M points, 3.11M samples [V] | Government-controlled. Only through a co-authored collaboration (CAS Nanjing Institute of Soil Science, CAAS). Out of scope for 2026–27. |
| China Soil Profile Dataset | 63k+ profiles (2nd survey) [V] | 25 printed volumes, not machine-readable |

### A4. Pseudo-labels: pretraining ablation only, never evaluation

gNATSGO/SSURGO, POLARIS, SoilGrids 2.0, OpenLandMap (**CC-BY-SA**), and the AI4SoilHealth EU data cube
(30 m SOC/pH/texture/BD predictions, open on Zenodo). These are model products. Training on them is how
the current model learned to copy OpenLandMap (lab SOC R² −0.44).

### A5. Deduplication and expected yield

Known overlaps:
- OSSL ⊃ KSSL, LUCAS
- WoSIS ⊃ NCSS
- ISCN ≈ re-packaged NRCS
- SoilGrids training points = WoSIS
- RaCA samples were run at KSSL
- WoSIS ⊃ parts of INEGI and CSDL

Dedup keys, in order:
1. Source keys (`pedon_key`, `labsampnum`).
2. Coordinates within 100 m **and** an overlapping depth interval.
3. Keep the record with the most complete method metadata.

Log every drop in `data/corpus/dedup_log.parquet`.

**Measured C1, 2026-11-16** (`geosoil/results/c1_summary.json`):

| | Count |
|---|---|
| Training sites, deduplicated | **190,567** (58,388 CONUS) |
| Hold-out sites (RaCA) | 6,367 |
| Lab values | 3.07M |
| Topsoil (<30 cm) training sites: SOC / pH / clay / sand / BD | 132.6k / 124.2k / 141.9k / 109.9k / 25.5k |
| Training sites by source | WoSIS 120.7k · KSSL 47.3k · Brazil 9.1k · Canada 8.1k · USGS 4.8k · OSSL 0.5k · NAPESHM 0.1k |
| Removed as cross-source copies | 8,300 WoSIS (of NPDB 4.5k, Brazil 2.8k, KSSL 1.0k), 1,589 OSSL (mostly in WoSIS) |
| Coordinate precision > 10 km | ~3.5k WoSIS sites (keep for probing, filter before embedding) |
| Training sites within 1 km of a RaCA hold-out site | 210 (flagged `near_holdout`) |
| Values under non-commercial licences | 18% (fine for this project; filter before any redistribution) |

**How dedup works:** a site is only dropped when its source is known to republish the other one (WoSIS and OSSL are
compilations). Independent surveys such as USGS DS 801 and NAPESHM are never dropped just for being near a KSSL pedon.
Native sources outrank their WoSIS copies.

**Next:** the A2 requests (~+100k if granted), then A3 collaborations (millions). Labels matter for probing and
fine-tuning. **Model size is set by the unlabelled corpus** (SCALING-PLAN.md).

## B. Unlabelled inputs: the pretraining corpus

All of these are point-sampled; the corpus never stores whole rasters.

### B1. Per-location inputs (one value or series per point)

| Modality | Source | Coverage | Access / cost | Cadence used |
|---|---|---|---|---|
| **AlphaEarth** embeddings | 64-d, 10 m, CC-BY 4.0 | global, **2017–2025** (bucket listing, 2026-11-16) [V]; 302,466 tiles of 8192², band-interleaved, 1024² zstd blocks | **Free, not requester-pays:** `https://data.source.coop/tge-labs/aef/v1/annual/` + `aef_index.parquet`. Bottom-up COGs: read the `.tiff` with its own transform, or the `.vrt` via `/vsis3/`. The GCS bucket is requester-pays. | annual |
| **TESSERA** embeddings | 128-d Matryoshka, 10 m, CC-BY (weights CC0) | global, **2024 only so far** [V]; 2017–24 planned | free, `geotessera` | annual |
| Climate | ERA5-Land (ARCO Zarr via Earth Data Hub, or CDS), ARCO-ERA5 (`gs://gcp-public-data-arco-era5`, anonymous), Daymet / gridMET / TerraClimate (Planetary Computer), PRISM | global / NA | free | daily → weekly patches |
| Vegetation, LST | MODIS 6.1 (Planetary Computer), HLS v2 (LP DAAC, us-west-2) | global | free | 8/16-day → weekly |
| Soil moisture | SMAP L3/L4 (`earthaccess`) | global 9 km | free | daily → weekly |
| Terrain | Copernicus DEM 30 m, 3DEP | global / US | free | static |
| **Soil minerals** | **EMIT L2B mineral ID + band depth**, 60 m [V] | dryland belt ±52° | free (LP DAAC) | static |
| **Radiometrics** | USGS NURE K/U/Th Bayesian grids [V]; Australia Radmap v4 (100 m, CC-BY) [V] | CONUS + AK; Australia | free | static. **Strong clay / parent-material predictor.** |
| Bare soil, SAR | S2 bare-soil composite (SYSI/BSC method), S1 RTC (Planetary Computer / AWS) | global | free in us-west-2 | annual |
| Land cover | ESA WorldCover, Dynamic World (EE only), CDL (US) | | free | annual |

**Coarse modalities are cheap.** ERA5-Land (9 km), SMAP (9 km) and MODIS (500 m) have far fewer unique
cells than we have points. Extract per unique cell, then join. Only the 10 m embeddings and S1/S2 are
truly per-point, and they are tiled.

**Measured AlphaEarth read cost** (`scripts/corpus/aef_read_cost.py`, 20 KSSL sites, 2020; results in
`geosoil/results/aef_read_cost_laptop.json`). Bytes are exact, from the TIFF block sizes. Seconds are from
a home connection, so expect Alpine to be faster.

| Pattern | Bytes | Seconds |
|---|---|---|
| One scattered point | ~38 MB per point-year (one whole 1024² block per band × 64 bands) | ~23 s |
| A dense block | ~37–38.5 B per pixel (1M pixels for ~40 MB) | ~20 s at home; **7.7 s on Alpine via direct S3** |

What that implies for each corpus:

| Corpus | Read volume | Verdict |
|---|---|---|
| 2M points × 9 years, scattered | ~684 TB | **Infeasible** |
| 1e9 pixel-years, dense | ~40 GB | **Cheap** |
| All 52,340 KSSL sites | They fall in only 18,963 distinct blocks (2.8 per block): 0.7 TB per year, 6.5 TB for 9 years | Feasible as streamed reads on Alpine |

**So every tier is built from whole blocks, never scattered points** (§C). Read from `s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/...`, not `data.source.coop`: from Alpine that is 66 MB/s against under 4 MB/s. One reader is capped near 5 MB/s by 64 serial band requests, so run many readers in parallel, or fetch the bands concurrently.

### B2. Pre-tiled ML datasets (alternative or extra pretraining data)

| Dataset | Size | Notes |
|---|---|---|
| TerraMesh | 9.09M samples, 7 modalities, 14 TB Zarr [V] | **CC-BY-SA** (share-alike) |
| Major TOM Core-S2L2A | 2.25M patches, ~23 TB [V]; plus embedding sets (DINOv2, SigLIP, UniverSat, MMEarth) | open, HF |
| MMEarth | 1.2M locations, 12 modalities incl. ERA5, DEM, land cover [V] | **no soil layer** |
| SSL4EO-S12 v1.1 | ~250k locations × 4 seasons [U] | |

These are patch datasets. GeoSoil is per-point, so they are a fallback or an image branch, not the core corpus.

### B3. Open pretrained models

| Model | Size | How it enters GeoSoil |
|---|---|---|
| **Galileo** (nasaharvest) | Nano 0.8M / Tiny 5.3M / Base 85M; pretrained on 127k samples × 24 months | Closest input match (S1, S2, ERA5, TerraClimate, SRTM, Dynamic World). Baseline + temporal-encoder initialization. |
| **TESSERA** | students 1M / 7M / 21M / 44M; 2B teacher (weights CC0) | Second embedding modality, baseline, and the distillation recipe |
| **Presto** | 0.4M; 21.5M pixel time series | Fast baseline |
| AlphaEarth | ~480M (weights not public) | Embeddings only |
| Prithvi-EO-2.0, TerraMind, AnySat, Copernicus-FM | patch / image models | Low priority |

Confirm the Galileo, Presto, AnySat and Copernicus-FM licenses from each repo before use.

## C. Sampling the unlabelled corpus

The goal is to cover as many distinct soil-forming conditions as possible per byte.

1. **Strata.** Climate (k-means on ERA5 normals), terrain class, land cover, and an AlphaEarth k-means
   cluster (k ≈ 256).
2. **The unit of sampling is a 1024² AlphaEarth block** (10.24 km square). Reading one costs the same
   as reading a single pixel in it (§B1), so every block read yields all of its pixels.
3. **Label-anchored blocks first.** Every block that contains a labelled site is read anyway to embed
   that site: about 19k blocks for KSSL alone, and ~300k-site C1 will touch more. Keep the site pixels
   plus thinned unlabelled pixels from the same block.
4. **Coverage blocks next.** Add blocks chosen by conditioned Latin hypercube sampling (cLHS; Minasny &
   McBratney 2006), or by iterative proportional fitting over the strata (Wickrema et al. 2025), until
   the strata are covered.
   - C2 (CONUS): ~2M locations ≈ 2k–20k blocks × 100–1,000 thinned pixels.
   - C3 (global): ~10M locations ≈ 10k–100k blocks.
   - C4: 1e9 pixel-years ≈ ~1k blocks × 9 years read in full, or more blocks thinned harder, which is
     better for diversity.
   Prefer **more blocks with fewer pixels each**: pixels inside one block are spatially autocorrelated.
5. **Thin near-duplicates within each block** by embedding distance, SemDeDup-style (arXiv 2303.09540).
   Prune C3/C4 hard and C2 lightly (Sorscher et al. 2022).
6. **Hold out** 1° blocks (~10%) and the latest year that both embedding sources cover. Fix the
   assignment once in `data/corpus/splits.parquet`.

## D. Choosing features

Start from SCORPAN and treat each candidate as a **group** (a modality), not a single column.

| SCORPAN factor | Candidate group | Current status |
|---|---|---|
| s (soil) | spectra (where present) | not built |
| c (climate) | ERA5-Land weekly series, climate normals, aridity index | 12-month version built |
| o (organisms) | MODIS phenology, AlphaEarth, TESSERA, land cover / CDL history | AlphaEarth + CDL built |
| r (relief) | elevation, slope, TWI, curvature, MRVBF | partly built |
| p (parent material) | **radiometrics, EMIT minerals**, surficial geology, bare-soil S2, S1 SAR | bare-soil + SAR built |
| a (age) | geologic age (geology map) | not built |
| n (space) | Fourier lon/lat | built |

**Protocol.** The evidence favours a hybrid of ~10–20 physical covariates plus the embeddings
(Koch et al. EGU26-13089; SpTGNN arXiv 2606.16580).

1. **Groupwise permutation importance and grouped SHAP, inside 5-fold spatial-block CV**, with LightGBM
   on lab labels. Importance from random CV rewards location proxies (Ploton et al. 2020). Report it
   per target.
2. **Drop-group ablation** for the neural model, via modality dropout.
3. **Keep a group** if it improves lab-label spatial-CV skill for at least one target by more than the
   seed noise, or widens the area of applicability. Drop it if it only helps on OpenLandMap labels.
4. **Temporal granularity test (R1):**
   - (a) climate normals + summary statistics
   - (b) monthly series
   - (c) weekly patches over 1–4 years
   Keep long series only where they beat (a).
5. **Validation reporting:** always report spatial-block CV, leave-region-out, and area-of-applicability
   masks. Add design-based accuracy where a probability sample exists (RaCA, USGS DS 801).

## E. Label harmonization

- **Methods.** Carry `method_id` for every label (WoSIS method codes, KSSL procedure codes). Use per-method
  offset heads rather than converting values up front.
- **SOC.** Walkley-Black vs dry combustion is not a flat 1.32: it ranges 0.9–1.5 and rises with clay
  (Shamrikova / FAO GSP; J. Plant Nutr. Soil Sci. 2025).
- **pH.** H₂O vs CaCl₂: use an EC-aware conversion (Minasny et al. 2011) or separate heads.
- **Depth.** Fit equal-area splines to 0–5, 5–15, 15–30, 30–60 and 60–100 cm.
- **Noise floor.** Estimate it from KSSL lab replicates per target, and store it with the results.

## F. Where else to look (unchecked leads)

Record each lead in `sources.yaml` with `status: lead`.

- **US:** USDA-ARS LTAR, EPA NARS wetland soils, USGS National Geochemical Database, CZNet, LTER, AmeriFlux site soils.
- **Data papers:** search Zenodo, PANGAEA, ESSD and Scientific Data for "soil profile database".
  Candidates include India's NBSS&LUP profiles, Russia's EGRPR, Argentina, Uruguay.
- **CU Boulder:** Greg Tucker's group, INSTAAR (Front Range / Critical Zone cores), CSDMS community data.
- **EU:** the AI4SoilHealth / EU Soil Observatory releases, and national databases beyond LUCAS.
- **Hugging Face / Kaggle:** search "soil". Check provenance and license before using anything.

Work items are numbered in EXECUTION-PLAN.md.
