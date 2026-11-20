# GeoSoil roadmap: an open soil foundation model

This is the entry point for continuing the work. It says where the project stands, what each phase
delivers, and **what has to be true before the next phase is worth starting**. The detail lives in
`docs/research/`:

| Doc | Use it for |
|---|---|
| [EXECUTION-PLAN.md](research/EXECUTION-PLAN.md) | Step-by-step checklist (steps 0.1–5.5), with status |
| [SCALING-PLAN.md](research/SCALING-PLAN.md) | How big a model each corpus supports, the compute it needs, overfitting guards |
| [CORPUS-PLAN.md](research/CORPUS-PLAN.md) | Every large soil dataset, how to get it, sampling, feature selection |
| [CURC-PLAN.md](research/CURC-PLAN.md) | Alpine partitions, limits, allocations, policy |
| [PIPELINE-DESIGN.md](research/PIPELINE-DESIGN.md) | How the data and feature pipeline is built |
| [LOCAL-MODELS.md](research/LOCAL-MODELS.md) | Model tiers (global, regional, edaphic zones, local) and how each is evaluated |
| [METRICS.md](research/METRICS.md) | How models are evaluated: why R² alone misleads, the metric suite, block-bootstrap CIs, paired tests |
| [DATA-REPORT.md](research/DATA-REPORT.md) | What data goes into the model, where it comes from, and what has proven useful |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Ground rules: public data, lab truth, licences, scrub check |

## Where things stand (2026-11-19, round 8)

Saved state for resuming; the detailed record is in DATA-REPORT, LOCAL-MODELS and METRICS.

**Model.** LightGBM on every feature group, 1° spatial-block CV, topsoil 0–30 cm, 341,969 training sites
from 26 sources plus 6,367 RaCA hold-out sites. Round-8 all-groups R²:

| Target | SOC | pH | Sand | Clay | BD | OM | CEC | K | Ca | Mg |
|---|---|---|---|---|---|---|---|---|---|---|
| R² | 0.564 | 0.621 | 0.470 | 0.409 | 0.525 | 0.567 | 0.391 | 0.452 | — | — |

(Ca and Mg were scored with the paired test only; round 7 was Ca 0.529, Mg 0.469.)

- **Round 8 vs round 7:** paired on shared rows, round 8 has lower error on all ten targets (p ≤ 0.011).
- **Bare soil is worth it:** the Sentinel-2 composite added clay +0.011 on unchanged rows.
- **Standard-method labels win:** training CEC, Ca and Mg on standard methods only lowers error further
  (p ≤ 0.002).
- Numbers are in `geosoil/results/round8_summary.json` and `scorecard_round7.json`.

**Findings that set direction:**
- **Evaluation.** R² only ranks models on identical rows. Use paired block tests, RMSE with block-bootstrap CIs,
  floor skill and kappa (METRICS.md, `geosoil/scorecard.py`).
- **Errors.** No bias, calibration slope about 1. All error is missing signal. Floor skill is 0.44–0.68.
- **Few-shot.** k lab samples in a ~5 km cell, used to update the global prior (empirical-Bayes offset):
  - k = 1 cuts error 10–23%; k = 5 cuts it 18–38%.
  - The samples alone never beat prior plus samples.
  - Code: `geosoil/fewshot.py`.
- **Local tier.** A 150 km regional blend lowers error 2–8% where a regional model can be fitted, and locality
  is confirmed against a random-subset control. Use global elsewhere (`geosoil/tiers.py`).
- **Coverage.** 18% of land is outside the area of applicability: India, Southeast Asia, Sahel, Arabia,
  Madagascar, the Andes (`geosoil/coverage.py`).
- **What data is worth.** New environments are worth 2–10× more than more of the same, and one standard-method
  label is worth 2–7 mixed ones. Priority: bases > BD > CEC > clay > sand > pH > SOC.

**Open items, in flight when saved:**
- **Earth Engine bare soil.** 18 READY tasks were left in the queue. Restart the collector on the laptop:
  `nohup scripts/bare_soil_collect_loop.sh > ~/.cache/geosoil-bare-soil-loop.log 2>&1 &`. Re-auth first if
  needed with `earthengine authenticate`. Then Globus the `bare_soil` directory to Alpine, using absolute
  laptop paths.
- **France RMQS.** Wired in, but the host was down. It is skipped automatically until reachable.
- **Data requests.**
  - CanSIS declined the full NPDB for now; a new release is coming and they invite gap reports.
  - ISRIC was re-sent to soils@isric.org.
  - ESDAC, iSDA and BSSL have not answered.
- **Restricted mode.** Earth Engine is in restricted mode (Community tier). Reopening a billing account for the
  Contributor tier is the biggest speedup.

## Next steps (ordered)

1. **Head-to-head with SoilGrids on RaCA.** RaCA is the probability-sample hold-out; use design-weighted RMSE.
   This settles the claim against the state of the art. SoilGrids is an evaluation baseline only, never a
   feature or label.
2. **Finish bare soil** for the last ~20% of sites, then rerun the round-8 scorecard.
3. **Standard-method heads for CEC/K/Ca/Mg.** Train on standard-method labels, with mixed-method labels as an
   auxiliary input or head. This has already been proven better by paired tests.
4. **Serving rule in code.** Use the 150 km blend wherever there are enough labels within the radius,
   otherwise global. Add the few-shot offset when a user supplies samples.
5. **Phase 3 pretraining.** A masked spatio-temporal encoder on the 39.7M unlabelled pixel-years, then
   multi-task fine-tuning. Gate: beat LightGBM on the few-shot curve at k = 1–3 and in the regions outside the
   area of applicability, not on the dense-CONUS average.
6. **Spectroscopy modality.** OSSL MIR/vis-NIR spectra paired with lab values, sharing one embedding with the
   remote-sensing encoder.
7. **Soil state over time.** Make year an explicit input and test whether the model captures change
   (SOC under management, pH after liming).
8. **Data, by coverage.**
   - LUCAS K via OSSL v1.3 (train-only).
   - China forest profiles (BD).
   - Tellus Ireland.
   - An ANSIS account (Australia).
   - Request drafts for SLU, Thünen, South Africa ARC, Morocco and Nepal.
   - A CanSIS gap report.
   - India and Southeast Asia collaborations once a v0.1 exists.
9. **Phosphorus** as an eleventh target (WoSIS has it).

## Improvement track (added 2026-11-17)

The round-2 baseline sits 0.3–0.5 R² below the noise-floor ceilings, and texture is furthest from its
ceiling. These five items are ordered by expected gain per unit of work. Each is measured with the same
protocol: same rows, same spatial folds, drop-group ΔR² across 3 seeds. An item is kept only if it adds
≥ +0.005 R² on at least one target without hurting the others.

| # | Item | Why | Status | Where |
|---|---|---|---|---|
| I1 | **Neighbouring-label features**: values and distances of the nearest training-fold sites | Soil is strongly spatially autocorrelated. Hengl et al. 2018 (RFsp) and most DSM studies find this the largest cheap gain. Must be built inside each fold, with training rows excluding their own spatial block so they see the same neighbour distances the test fold does. | **built**; local check vs location+depth alone: SOC 0.33→0.40, pH 0.27→0.45, sand 0.13→0.31, clay 0.12→0.22, BD 0.23→0.33. Full test on top of all features: `NEIGHBOURS=1 sbatch scripts/slurm/baseline_c1.sbatch` | `geosoil/features/neighbours.py` |
| I2 | **C1 neural trainer** (step 2.6) | The old `train.py` targets the first OpenLandMap pipeline. A tabular network will roughly tie LightGBM; its value is the path to pretraining on the 39.7M unlabelled AlphaEarth pixel-years and the per-year sequence. | **built** (residual MLP, multi-target masked loss, 3 seeds, with and without neighbours); run with `sbatch scripts/slurm/train_c1.sbatch` (H200 slice) | `geosoil/train_c1.py` |
| I3 | **Spatial context for AlphaEarth**: embedding means and spread in windows around each site | Multi-scale terrain earned its place the same way; one pixel misses the surrounding landscape. | **running**: 91 Earth Engine exports, 500 m and 3 km context plus site-vs-context similarity; trial 100% coverage | `geosoil/features/gee_context.py` |
| I4 | **More texture labels outside the US** | Non-US sites lack radiometrics and surficial maps, and sand/clay are furthest from their ceilings. | **SISLAC ingested** (+28,190 Latin American training sites; C1 now 218,757). Verified next: SODES (Spain), Queensland SALI (~4k), German BZE-LW (2.9k). LUCAS/iSDA/BSSL/WoSIS requests pending | `geosoil/corpus/sislac.py`, `data/corpus/sources.yaml` |
| I5 | **Bare-soil reflectance composite** (Sentinel-2, multi-year, vegetation masked; Demattê et al. 2018, 2020) | The strongest remote texture signal in the literature, and global. Needs Earth Engine, which the project now has (`geo-soil-foundational-model`). | **running**: 91 Earth Engine exports (79 + a 12-task top-up for the SISLAC sites); trial: 84% of sites have bare-soil reflectance | `geosoil/features/gee_context.py` |

**Earth Engine.** The project `geo-soil-foundational-model` is registered for noncommercial use. Auth
locally with `gcloud auth login` (or `earthengine authenticate`); the extractors take a token from
gcloud. Sampling runs as batch exports to table assets under `projects/geo-soil-foundational-model/assets/geosoil`
(interactive calls time out on multi-year composites), four at a time per project:
`python -m geosoil.features.gee_context --part <bare_soil|alphaearth_context> submit|collect`. Submits are
incremental: new labels export only their own sites.

**After the Earth Engine exports and new labels land, rerun on Alpine:** `corpus_build.sbatch` (C1 with
SISLAC; needs `bsdtar`), AlphaEarth/terrain/static extraction for the 28k new sites, copy the two
`data/processed/features/{bare_soil,alphaearth_context}` directories over, then `NEIGHBOURS=1
baseline_c1.sbatch`, `feature_selection.sbatch` and `train_c1.sbatch`.

## Phases

The model size per phase comes from the TESSERA v2 compute-optimal law, N ≈ 2.06e9·(D/1.4e10)^0.571,
where D is unique location-years. The **rung gate** decides whether a bigger model actually helped: at least
+0.02 R² on held-out lab labels, across 3 seeds.

| Phase | Goal | Model | Main deliverables | Gate to start the next phase |
|---|---|---|---|---|
| **0. Access + baseline** | Prove the stack runs on Alpine | presets 0.4–31M | env, GPU smoke test, baseline sweep on KSSL lab labels | Smoke test passes; baseline numbers recorded |
| **1. Labels** ✅ | Every openly pullable lab label, clean | — | `labels_c1`, `labels_harmonized`, `splits`, `noise_floor` | ≥100k unique sites (have 190k) |
| **2. Features + R1** | Inputs for every labelled site, first real model | **~30M** | block-based AlphaEarth, terrain, climate, radiometrics feature store; feature-selection report; R1 bake-off vs Galileo / TESSERA / Presto embeddings | **R1 beats the best frozen open-model embedding + linear probe on lab labels** (spatial CV) |
| **3. CONUS corpus + R2** | First unlabelled pretraining | **~100M** | ~2M CONUS locations from stratified blocks; pretraining; first scaling fit | R2 ≥ R1 + 0.02 R² (3 seeds); otherwise grow data, not the model |
| **4. Global corpus + R3 + v0.1** | Global model, first public release | **~250M** → distilled 20–50M | ~10M global locations; distillation; eval suite; v0.1 weights + embeddings | R3 ≥ R2 + 0.02 R² **and** an allocation in hand |
| **5. Dense corpus + R4 + v1.0** | The large model | **0.5–1B** teacher → 20–50M student | ~1e9 location-years from dense blocks; teacher; distilled student; paper | — |

**What each phase costs:**

| Phase | Compute | Feature store |
|---|---|---|
| 2 | minutes of GPU; ~150 reader-hours of extraction per AlphaEarth year | ~5 GB |
| 3 | ~7 A100-h | ~30 GB |
| 4 | ~80 A100-h | ~140 GB |
| 5 | ~1,400–4,300 A100-h (Ascent covers ~650M; Peak or ACCESS Discover for 1B) | ~1.7 TB in `/scratch` |

## What unlocks progress

Each resource below opens a specific part of the roadmap. When one arrives, start with the step in the
right-hand column.

| When you get… | It unlocks | Start with |
|---|---|---|
| A free A100 slot | Phase 0 completion | Read the `logs/bootstrap_*.out` result, then run `scaling_sweep.sbatch` |
| **A faculty PI** (Greg Tucker first) for Ascent, or an advisor as co-PI for ACCESS | Phase 5, and faster queues for everything (Ascent outranks Trailhead) | CURC-PLAN "Getting more"; apply with the R1/R2 job numbers |
| Replies from **iSDA / BSSL** | +50–100k sites in Africa and Brazil, which are badly under-represented | Add a `geosoil/corpus/<source>.py` loader (same pattern as `brazil.py`), then rerun `build_c1` |
| Replies from **ISRIC / CanSIS** | Restricted WoSIS profiles; full Canadian pedons | Same loader pattern |
| **LUCAS 2022** (public ~Q1 2027) | KSSL → EU transfer test | `geosoil/corpus/lucas.py`; never commit the rows |
| A **collaboration** (ICAR-IISS / IIT Delhi for India SHC; CAS Nanjing for China) | Millions of labels | Only after v0.1 exists, so there is something to offer |
| **More time** (semester break) | Phases 2–3 back to back | Feature store first: models are cheap once it exists |
| **More storage** (beyond the 250 GB `/projects`) | Phase 5 feature store | Keep it in `/scratch` with a re-runnable build (90-day purge) |

## Standing rules

- **Public data only; open results.** No commercial or employer data, code or credentials, in this repo or
  on Alpine. Architecture patterns learned elsewhere are fine; implementations are written fresh.
- **Lab-measured truth is the target.** Model products (OpenLandMap, SoilGrids, POLARIS) are never evaluation labels.
- **Choose checkpoints by spatial-CV probe skill, not pretraining loss.**
- **Never report above the noise floor**; treat skill near it as suspected leakage.
- **Register a source** in `data/corpus/sources.yaml` (with its licence) before any code downloads it.

## How to resume after a break

```bash
git pull
python -m pytest -q tests                      # everything still green?
python -m geosoil.corpus.build_c1              # labels (downloads are cached in data/raw/)
cat geosoil/results/c1_summary.json            # current corpus size
grep -n "✅\|^| [0-9]" docs/research/EXECUTION-PLAN.md | head -60   # next unticked step
```

On Alpine, use the OnDemand Jupyter terminal:

```bash
cd /projects/$USER/soil-state-transformer && git pull
sbatch scripts/slurm/alpine_env_build.sbatch   # only if the env is missing
nohup scripts/slurm/watch_jobs.sh <ntfy-topic> <ids-file> &   # follow jobs remotely
```
