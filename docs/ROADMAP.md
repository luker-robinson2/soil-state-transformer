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
| [DATA-REPORT.md](research/DATA-REPORT.md) | What data goes into the model, where it comes from, and what has proven useful |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Ground rules: public data, lab truth, licences, scrub check |

## Where things stand (2026-11-16)

**Done:**
- **Labels (C1).** 190,567 deduplicated lab-measured training sites (58k CONUS) plus 6,367 RaCA hold-out
  sites, from 8 sources pulled without accounts. They are harmonized to standard depths, split into fixed
  spatial folds, and have a measured noise floor (topsoil R² ceilings 0.82–0.92).
  Rebuild: `python -m geosoil.corpus.build_c1`.
- **Alpine.**
  - Access works through the OnDemand Jupyter terminal; SSH is still broken.
  - The conda env is built (`ENV OK`) and the GPU smoke test passed on an H200 MIG slice (11-17). GPU caps
    are per type: when one pool shows QOSGrpGRES, submit to another (`--partition=ah200 --gres=gpu:h200_3g.71gb:1`).
  - Compute nodes have internet.
  - AlphaEarth reads come straight from S3 (66 MB/s).
- **Measured design constraint.** AlphaEarth must be read a whole 1024² block at a time (37 B per pixel),
  never as scattered points (38 MB per point). Every corpus tier is built from blocks.

**C1 baseline** (LightGBM, spatial CV, lab labels; all feature groups incl. multi-scale terrain, lithology, LST, HAND, land-use history): SOC R² 0.55, pH 0.60, sand 0.45, clay 0.39, BD 0.52. Ceilings are 0.82–0.92. This is the bar R1 must beat (`geosoil/results/baseline_c1.json`).

**In progress:**
- Phase 2 features. Done: AlphaEarth 2017–2025 and terrain for all sites. Built and ready to run: climate normals and radiometrics (`bash scripts/slurm/run_phase2_static.sh <topic>` from the Alpine terminal). Next: the feature-selection report, then the R1 neural bake-off on GPU.
- Data requests sent 2026-11-16 to iSDA, BSSL (Demattê), ISRIC, CanSIS and ESDAC.

## Improvement track (added 2026-11-17)

The round-2 baseline sits 0.3–0.5 R² below the noise-floor ceilings, and texture is furthest from its
ceiling. These five items are ordered by expected gain per unit of work. Each is measured with the same
protocol: same rows, same spatial folds, drop-group ΔR² across 3 seeds. An item is kept only if it adds
≥ +0.005 R² on at least one target without hurting the others.

| # | Item | Why | Status | Where |
|---|---|---|---|---|
| I1 | **Neighbouring-label features**: values and distances of the nearest training-fold sites | Soil is strongly spatially autocorrelated. Hengl et al. 2018 (RFsp) and most DSM studies find this the largest cheap gain. Must be built inside each fold, with training rows excluding their own spatial block so they see the same neighbour distances the test fold does. | in progress | `geosoil/features/neighbours.py` |
| I2 | **C1 neural trainer** (step 2.6) | The old `train.py` targets the first OpenLandMap pipeline. A tabular network will roughly tie LightGBM; its value is the path to pretraining on the 39.7M unlabelled AlphaEarth pixel-years and the per-year sequence. | planned | `geosoil/train_c1.py` |
| I3 | **Spatial context for AlphaEarth**: embedding means and spread in windows around each site | Multi-scale terrain earned its place the same way; one pixel misses the surrounding landscape. | in progress (Earth Engine) | `geosoil/features/gee_context.py` |
| I4 | **More texture labels outside the US** | Non-US sites lack radiometrics and surficial maps, and sand/clay are furthest from their ceilings. | researching pullable sources; LUCAS/iSDA/BSSL requests pending | `data/corpus/sources.yaml` |
| I5 | **Bare-soil reflectance composite** (Sentinel-2, multi-year, vegetation masked; Demattê et al. 2018, 2020) | The strongest remote texture signal in the literature, and global. Needs Earth Engine, which the project now has (`geo-soil-foundational-model`). | in progress (Earth Engine) | `geosoil/features/gee_context.py` |

**Earth Engine.** The project `geo-soil-foundational-model` is registered for noncommercial use. Auth
locally with `gcloud auth login` (or `earthengine authenticate`); the extractors take a token from
gcloud. Sampling is server-side, so these extractors run from a laptop or Alpine alike.

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
