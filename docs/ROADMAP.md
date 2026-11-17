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
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Ground rules: public data, lab truth, licences, scrub check |

## Where things stand (2026-11-16)

**Done:**
- **Labels (C1).** 190,567 deduplicated lab-measured training sites (58k CONUS) plus 6,367 RaCA hold-out
  sites, from 8 sources pulled without accounts. They are harmonized to standard depths, split into fixed
  spatial folds, and have a measured noise floor (topsoil R² ceilings 0.82–0.92).
  Rebuild: `python -m geosoil.corpus.build_c1`.
- **Alpine.**
  - Access works through the OnDemand Jupyter terminal; SSH is still broken.
  - The conda env is built (`ENV OK`); the GPU smoke test waits on the shared A100 pool.
  - Compute nodes have internet.
  - AlphaEarth reads come straight from S3 (66 MB/s).
- **Measured design constraint.** AlphaEarth must be read a whole 1024² block at a time (37 B per pixel),
  never as scattered points (38 MB per point). Every corpus tier is built from blocks.

**In progress:**
- Phase 2 feature extraction (`geosoil/features/`).
- Data requests sent 2026-11-16 to iSDA, BSSL (Demattê), ISRIC, CanSIS and ESDAC.

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
