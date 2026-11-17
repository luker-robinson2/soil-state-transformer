# GeoSoil foundation model: step-by-step execution plan

Draft 2026-11-16. This is the "what to do next, in order" companion to:
- CURC-PLAN.md: cluster
- CORPUS-PLAN.md: data
- SCALING-PLAN.md: model sizes

Dates are targets, set around the semester (STAT 5010 final project due 12-04, finals mid-December,
winter break until mid-January). Each phase ends in a **gate**: a measured result that decides whether
the next phase is worth doing.

Legend:
- **Who:** L = Luke (needs his accounts, signature or browser); C = Claude (code, drafts, analysis);
  job = an Alpine batch job.
- **Where:** laptop, `acpu` (CPU partition), `aa100` / `ah200` (GPU partitions), `dtn` (data transfer node).

---

## Phase 0: access, baseline, and start the slow requests (11-16 → 11-27)

The long pole is data requests (weeks to months), so they go out first.

| Step | Action | Who | Where | Done when |
|---|---|---|---|---|
| 0.1 | Review and merge PR #1 (plans + fixed Slurm scripts) | L | GitHub | merged to `main` |
| 0.2 | Fix SSH (rc-help ticket) or keep using the OnDemand Jupyter terminal | L | browser | either works |
| 0.3 | Paste `scripts/slurm/alpine-paste.sh` into the Jupyter terminal → bootstrap job | L → job | `aa100` | log ends `BOOTSTRAP OK`; the internet line says yes or no |
| 0.4 | Copy `data/processed/gee_*.parquet` to Alpine; rebuild master.parquet | L/C | laptop → `dtn` | `modality_presence` shows climate, veg, moisture = True |
| 0.5 | R0 sweep: `sbatch scripts/slurm/scaling_sweep.sbatch` (4 sizes × 2 label sources) | job | `aa100` | 8 results JSONs; a size-vs-skill table on OpenLandMap **and** KSSL |
| 0.6 | ✅ 11-16, laptop: AlphaEarth read cost measured (`scripts/corpus/aef_read_cost.py`). Scattered ≈ 38 MB / 23 s per point-year; dense ≈ 38.5 B per pixel. KSSL's 52k sites fall in 19k blocks (6.5 TB for 9 years). **Decision: block-based corpus.** **Alpine rerun 11-16** (`aef_read_cost_alpine.json`): compute nodes have internet. Direct S3 HTTPS reaches 66 MB/s; the data.source.coop proxy managed under 4 MB/s, so read S3 directly. One block (1M pixels, 37 MB) takes 7.7 s per reader because of 64 serial band requests, so readers scale in parallel. Embedding all KSSL sites for 9 years (171k block reads) is ~365 serial hours, ~6 h with 64 readers. Still to do: check TESSERA's years. | C → job | `acpu` | `geosoil/results/aef_read_cost_laptop.json` |
| 0.7 | **Send data requests** (C drafts, L sends): iSDA full table, Brazil BSSL (Demattê), ISRIC restricted WoSIS, Canada NPDB full, ESDAC LUCAS 2022 registration, TERN API key | C → L | email | 6 requests sent; each logged in `sources.yaml` as `status: requested` with the date |
| 0.8 | Ask a faculty member (Greg Tucker first) to be PI on Ascent or co-PI on ACCESS. Ask rc-help whether a grad student can be Ascent PI. | L | email | yes from one faculty member, or the rc-help answer |
| 0.9 | Earth Engine noncommercial project: Contributor tier needs a billing account. Decide whether it's needed at all, given the Source Cooperative mirror. | L | console | decided |

**Gate 0:** the bootstrap passes, the R0 table exists, and step 0.6 says how many points per hour we can
extract. If compute nodes have no internet, every download step below runs on `dtn` or as staged copies.

## Phase 1: labelled corpus C1 (11-23 → 12-13)

Label tables are small (GB), so most of this runs on the laptop or a single `acpu` job. Each step is one
PR with a test.

| Step | Action | Who | Output |
|---|---|---|---|
| 1.1 | ✅ 11-16: label schema (long format, one row per value per method) | C | `geosoil/corpus/schema.py` |
| 1.2 | ✅ 11-16: KSSL ingest (`data/extraction/kssl_export.R` → `python -m geosoil.corpus.kssl`), 9 tests. From the 2026-07 labtaxa mirror (USDA LDM was down): **52,340 georeferenced pedons (50,008 CONUS), 1.51M values**. Topsoil sites: SOC 41.8k, pH 41.4k, sand 49k, clay 50k, BD 15.9k. In KSSL, Walkley-Black ≈ dry combustion (median ratio 0.98, r = 0.94). Rerun from the official GeoPackage when the LDM site is back. | C | `labels_kssl.parquet` |
| 1.3 | ✅ 11-16: WoSIS 2023 (`geosoil/corpus/wosis.py`). 152.7k sites after dropping US-NCSS. Licence kept per row rather than filtered, since non-commercial use is fine here. | C | `labels_wosis_2023.parquet` |
| 1.4 | ✅ 11-16: Brazil SoilData 2024 (9.2k sites), Canada NPDB (8.1k), NAPESHM (1.5k plots), USGS DS 801 (4.8k). ISCN3, INEGI and CSDLv2 are not pullable (see `sources.yaml`). | C | one parquet per source |
| 1.5 | ✅ 11-16: OSSL v1.2 lab values without KSSL/LUCAS (2.1k sites). Spectra not pulled yet. | C | `labels_ossl_v1_2.parquet` |
| 1.6 | ✅ 11-16: RaCA hold-out (6.4k sites, 33k lab SOC, 82k measured BD; ~1 km coordinates). None of its lab samples appear in KSSL. **Doubles as the STAT 5010 final-project data (due 12-04).** | C + L | `labels_raca.parquet` |
| 1.7 | LUCAS: LUCAS-MEGA isn't shared by its authors. LUCAS 2022 waits on the ESDAC reply (public ~Q1 2027). | C | local parquet, gitignored |
| 1.8 | ✅ 11-16 (`geosoil/corpus/harmonize.py`): thickness-weighted averaging onto 0–5 / 5–15 / 15–30 / 30–60 / 60–100 / 100–200 cm, keeping an interval only at ≥50% coverage. One method per target, conversions fitted on KSSL paired layers: pH CaCl₂→H₂O (n 212k, R² 0.96); Walkley-Black→dry combustion ×1.019 (n 8.8k; KSSL Walkley-Black is already corrected); oven-dry→third-bar BD (n 78k, R² 0.69). `method_family` and `converted` are kept so a model can learn offsets. 2.49M rows. An equal-area spline is a possible later upgrade. | C | `labels_harmonized.parquet`, `results/harmonization.json` |
| 1.9 | ✅ 11-16: dedup (`geosoil/corpus/dedup.py`) and the one-shot builder `python -m geosoil.corpus.build_c1`. **C1 = 190,567 training sites (58k CONUS) + 6,367 hold-out; 3.07M values.** 25 tests. | C | `labels_c1.parquet`, `dedup_log.parquet`, `results/c1_summary.json` |
| 1.10 | ✅ 11-16 (`geosoil/corpus/splits.py`): 1° blocks hashed, so assignments stay stable as data grows. Train 176,388 sites in 5 folds (31k–40k each), test 14,179 (1 in 10 blocks), RaCA hold-out 6,367; `near_holdout` carried through. Temporal hold-out year for satellite inputs: 2024. | C | `splits.parquet`, `results/splits_summary.json` |
| 1.11 | ✅ 11-16 (`geosoil/corpus/noise_floor.py`): KSSL publishes no replicates in our tables, so the floor comes from distinct pedons within 30 m at the same depth and method (lab error + sub-30 m variation). Topsoil R² ceiling: SOC 0.87, pH 0.90, clay 0.87, sand 0.92, BD 0.82 (3.7k–14.6k pairs). Placeholder coordinates (more than 3 pedons on one point) are skipped. | C | `results/noise_floor.json` |

**Gate 1:** the real deduplicated count is known (estimate ~300k global). If it's under 100k, push the
§A2 requests harder before scaling the model.

## Phase 2: feature store C1, feature selection, R1 (12-07 → 01-10)

| Step | Action | Who | Where | Output |
|---|---|---|---|---|
| 2.1 | 🔄 11-17: **AlphaEarth 2017–2025 extracted for all C1 sites**: 1.65M site-years, **0 failed blocks in all 9 years** (three DNS-hit 2024 shards were re-run with the new retry logic, 0 failures), and **39.7M unlabelled pixel-years** kept as the Phase 3 seed. Terrain also extracted for all 196k sites. Climate (TerraClimate normals) and radiometrics (NURE + Radmap) extractors built on a shared raster sampler, 11-17; live-checked at Boulder (K 1.73 %, Jan tmax 8.1 °C), ready to run with `scripts/slurm/run_phase2_static.sh`. Extractor built and validated (`geosoil/features/alphaearth.py`; pixel values identical to direct reads; 5 tests). C1 plan: 183,681 sites × 2017–2025 = 1.65M site-years in 623,920 blocks. Design in PIPELINE-DESIGN.md. Remaining: point extractors, one module per source, all writing to one Zarr feature store keyed by `site_id`. Sources: ERA5-Land, MODIS, SMAP, Copernicus DEM / 3DEP, NURE radiometrics, EMIT minerals, S1, bare-soil S2, WorldCover/CDL, AlphaEarth, TESSERA. | C | `acpu` array (per source × chunk) | `features/c1/*.zarr` |
| 2.2 | Unique-cell extraction for coarse grids (ERA5, SMAP, MODIS) → join to points | C | `acpu` | fewer reads than points |
| 2.2b | ✅ 11-16: **first C1 baseline** (`geosoil/baseline_c1.py`, LightGBM, 5-fold spatial CV, topsoil, AlphaEarth 2020 + Copernicus terrain). Lab-label R², with the ceiling in brackets: **SOC 0.53 (0.86), pH 0.57 (0.90), sand 0.37 (0.92), clay 0.32 (0.87), BD 0.48 (0.82)**. **11-17, multi-year AlphaEarth mean (8 years, 2024 held out): SOC 0.54, pH 0.58, sand 0.40, clay 0.34, BD 0.49**, a consistent +0.01–0.02 on every target. Before C1, the best lab SOC was ~0.13–0.22 in-domain and −0.44 in transfer. Texture is weakest, so radiometrics and parent material come next. | C | `acpu` | `results/baseline_c1.json` |
| 2.3 | 🔄 11-17: feature-selection report built (`geosoil/feature_selection.py`): grouped permutation ΔR², SHAP share and drop-group ablation per target inside the fixed spatial folds; queued after the climate/radiometrics baseline (`scripts/slurm/feature_selection.sbatch`). | C | `acpu` | `results/feature_selection.md` |
| 2.4 | Temporal granularity test: normals vs monthly vs weekly patches | job | `aa100` | table |
| 2.5 | External baselines: AlphaEarth, TESSERA, Galileo, Presto embeddings + linear probe on C1 | job | `aa100` | baseline table |
| 2.6 | Model code: patch embedding, Transformer backbone, method heads, modality dropout, checkpoint/requeue, DDP | C | laptop + `gpu-testing` | PRs with tests |
| 2.7 | **R1** (~30M): GRU / Transformer / Mamba-2 at equal FLOPs; Galileo init vs scratch; 3 seeds | job | `aa100` | R1 report |
| 2.8 | NAIRR application (monthly deadline the 15th; aim for 01-15) and/or Ascent with R1 job numbers | L (C drafts) | web | submitted |

**Gate 2:** R1 beats the best external baseline (frozen embeddings + probe) on lab labels. If it
doesn't, fix features and objectives before adding data. Scale amplifies a broken setup.

## Phase 3: CONUS corpus C2 + R2 (01-11 → 02-07)

| Step | Action | Who | Where | Output |
|---|---|---|---|---|
| 3.1 | Block sampler: label-anchored blocks + cLHS coverage blocks, then within-block embedding thinning → ~2M CONUS locations | C | `acpu` | `blocks_c2.parquet` |
| 3.2 | Run the extractors at 2M points × 8 years | job | `acpu` array (+ `dtn`) | `features/c2` (~30 GB) |
| 3.3 | **R2** (~100M) pretraining, 4 epochs max | job | `ah200` or `aa100` | checkpoints |
| 3.4 | Probe R1 vs R2 on C1 labels; first scaling fit (probe skill vs N and D) | C | `aa100` | the scaling plot |

**Gate 3 (the rung gate):** R2 beats R1 by at least 0.02 R² (lab SOC or the 5-target mean, 3 seeds).
- If yes, go up a tier.
- If no, the model is data-limited: improve the corpus (features, sampling) before going bigger.

## Phase 4: global corpus C3 + R3 + first distilled release (02-08 → 03-21)

| Step | Action | Who | Where | Output |
|---|---|---|---|---|
| 4.1 | Global block sampler → ~10M locations (needs global coverage in every extractor) | C | `acpu` | `blocks_c3.parquet` |
| 4.2 | Extraction at 10M × 8 years | job | `acpu` array | `features/c3` (~140 GB) |
| 4.3 | **R3** (~250M), about 8 h on 4×H200 | job | `ah200` | checkpoints |
| 4.4 | Distill R3 → 20–50M Matryoshka student | job | `aa100` | student weights |
| 4.5 | Evaluation suite: spatial CV, leave-region-out, KSSL → LUCAS, US → Brazil, RaCA hold-out, area of applicability | C | `aa100` | eval report |
| 4.6 | First public release: student weights + embeddings for labelled sites + model card (CC-BY, data licenses permitting) | L + C | HF / GitHub | v0.1 release |

**Gate 4:** R3 passes the rung gate over R2, and an allocation (Ascent, Peak or ACCESS) is in hand.

## Phase 5: dense corpus C4 + the large model R4 (03-22 → 05-15)

| Step | Action | Who | Where | Output |
|---|---|---|---|---|
| 5.1 | Extend the C3 blocks to ~1e9 pixel-years (more blocks, thinned); stream each block once (~40 MB) | job | `acpu` array (+ `dtn`) | `features/c4` (~1.7 TB in `/scratch/alpine`; rerunnable, 90-day purge) |
| 5.2 | **R4a** (~650M, 2 epochs, ~420k SU budget) on Ascent | job | `ah200` (4×H200, 24 h jobs with requeue) | teacher checkpoints |
| 5.3 | If R4a passes the rung gate over R3 and allocation allows: **R4b** (~1B, 4 epochs) | job | `ah200` | teacher |
| 5.4 | Distill the teacher → 20–50M student; cache embeddings for CONUS / global | job | `aa100` | student + embedding product |
| 5.5 | Paper + v1.0 release (weights, embeddings, corpus build scripts, eval suite) | L + C | | v1.0 |

## Running in parallel throughout

- **Data requests (§A2):** check replies weekly and ingest each grant as a Phase 1 step.
- **Collaborations (§A3):** India SHC (ICAR-IISS / IIT Delhi) and China (CAS Nanjing) are long shots.
  Raise them only once v0.1 exists, so there's something to offer.
- **Coursework:** STAT 5010 (RaCA, 12-04); CSCI 5622 public-data parts.
- **Policy check before every push:** the scrub grep in CONTRIBUTING.md.

## What Claude can start immediately

1. Step 0.7: draft the six data-request emails.
2. Step 1.1–1.2: label schema + KSSL Lab Data Mart ingest.
3. Step 0.6: the AlphaEarth read-cost measurement script, to run as an `acpu` job.
