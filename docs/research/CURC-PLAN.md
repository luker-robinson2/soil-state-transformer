# GeoSoil on CU Research Computing (Alpine): plan

Written 2026-11-16; revised the same day after checking every fact against curc.readthedocs.io,
access-ci.org, nairrpilot.org and the Earth Engine docs. Companion docs: SCALING-PLAN.md (model and
compute ladder), CORPUS-PLAN.md (data sources and acquisition), EXECUTION-PLAN.md (step-by-step schedule).

## Where GeoSoil stands

**Model.** The four presets in `geosoil/config.py`, as instantiated (GRU temporal encoder, all modalities on):

| Preset | Trainable params |
|---|---|
| small | 0.43M (0.48M with temporal branches) |
| base | 1.5M (1.6M with temporal branches) |
| large | 9.8M (10.1M with temporal branches) |
| xl | 31M (32M with temporal branches) |

Each preset also carries an EMA target copy for JEPA, which is not trained.
One A100, or even one MIG slice, is enough for every current run.

**Labels on disk** (all in git):

| File | Rows | Has AlphaEarth features? |
|---|---|---|
| `soil_unified.csv` | 3,000 OpenLandMap points | yes |
| `kssl_with_features.csv` | 2,983 KSSL lab pedons | yes |
| `wosis_conus.csv` | 37,311 WoSIS lab profiles | no, labels only and unused so far |

**Modalities.** The GEE extracts (`gee_static/temporal/soil/dynamics.parquet`) are gitignored, so they
never reached Alpine:
- They hold terrain, ERA5 / MODIS / SMAP **12-month** sequences, bare-soil S2, S1 SAR, precipitation events and CDL.
- `cv_jepa.json` shows all of them were present for the OpenLandMap run.
- A fresh clone on Alpine therefore builds a master.parquet with **no temporal modalities**, so `--variant mamba` has nothing to encode there.
- Either copy the parquets over, or rerun `data/extraction/extract_gee*.py`.

**Read the scores by label source.** These come from 5-fold spatial-block CV, from `geosoil/results/` and the white paper:

| Label source | SOC R² | pH R² | Notes |
|---|---|---|---|
| OpenLandMap (a model product, not a measurement) | 0.76 | 0.93 | all modalities |
| OpenLandMap | 0.67 | 0.90 | the 11-16 CPU `--quick` check |
| KSSL lab truth, model trained on OpenLandMap | **−0.44** | 0.58 | `verification_jepa.json` |
| KSSL lab truth, in-domain (best of Ridge / LightGBM / +SAR/bare-soil) | about 0.13–0.23 | 0.53–0.55 | |

The OpenLandMap numbers mostly measure how well we copy another model. The lab numbers are the real
target, and they are low.

**The bottleneck is data, not compute:**
1. Lab-measured labels, which we have at about 43k CONUS rows on disk and roughly 300k globally available.
2. Covariates that physically sense soil (radiometrics, bare-soil reflectance, SAR, parent material).
3. Multi-year daily climate and vegetation series, long enough for a sequence model to matter.

CORPUS-PLAN.md turns this into a build plan.

## Compute you have right now (Trailhead / `ucb-general`)

Trailhead is free and never expires. It is a fair-share pool, about 20% of UCB's share of Alpine,
shared by every UCB user. Heavy use lowers your fair-share priority, but when the queue is idle your
job runs regardless of past usage.

There is **no SU cap**. What limits you is the per-user concurrency caps plus queue wait.

| Partition | Per node | Per-user cap (gpu-normal, 24 h) | gpu-long cap (7 d) | SU per GPU-hour |
|---|---|---|---|---|
| `aa100` | 3× A100 40 GB (some 80 GB), 64 cores, 243 GB RAM | 6× A100-40 | 3 | 108.6 |
| `ah200` | 4× H200 141 GB, 128 cores, 1.5 TB RAM, 5.4 TB local SSD | 4 | 2 | 370.4 |
| `artxpro6000` | 4× RTX Pro 6000 96 GB, 128 cores, 1.5 TB RAM, 5.4 TB local SSD | 4 | 2 | 260.4 |
| `al40` | 3× L40 48 GB | | | 108.6 |
| `acpu` (renamed from `amilan`, Aug 2026) | 32–128 cores, 3.8 GB per core | QoS `cpu-normal` / `cpu-long` | | 1 per core-hour |

Notes on the table:
- **CPU billing on GPU jobs.** Each core also bills 1 SU/h. A GPU job can request at most 21 cores per A100 and 32 per H200 or RTX GPU.
- **MIG slices** cut cost for small models:
  - `a100_3g.20gb`
  - `h200_2g.35gb`: 123.5 SU/h, 6 per user
  - `rtx_pro_6000_1g.24gb`: 65.1 SU/h, 7 per user. Request it with `--gres=gpu:rtx_pro_6000_1g.24gb:1`.
- **Testing.**
  - `gpu-testing`: 1 h, billed at 10%, 1 GPU, 5 jobs. **Only on `aa100` (the `a100_3g.20gb` slice) and `ami100`.**
  - `cpu-testing` on `acpu`: 2 nodes or 16 cores, 1 h.
  - `atesting` and the old `testing` QoS were removed in Oct 2026.
- **Interconnect.** No GPU node has InfiniBand: A100 nodes use 2×25 GbE RoCE, H200/RTX nodes use 100 GbE. Stay on one node. The largest single node is 4× H200 (564 GB of GPU memory). An A100 node holds only **3** GPUs.
- **Storage:**

  | Location | Limit | Notes |
  |---|---|---|
  | `/home` | **2 GB** | Never put the conda environment here; see below. |
  | `/projects/$USER` | 250 GB | |
  | `/scratch/alpine` | 10 TB and 20M files | Deleted 90 days after **creation**. |
  | `$SLURM_SCRATCH` (node-local SSD) | 5.4 TB on H200/RTX nodes, 416 GB on older nodes | Wiped when the job ends. |

- **Software.**
  - Use `module load miniforge`; it is the supported conda.
  - RTX Pro 6000 needs CUDA >12.8 and H200 needs >12.0. `environment-cuda.yml` pins torch 2.4.1+cu121: fine on A100, but not on RTX Pro.
  - The login nodes throttle heavy work (Arbiter2), so build in `acompile` or inside a job.
- **Data transfer.** Use `dtn.rc.colorado.edu` (scp/rsync/sftp), or Globus collection "CU Boulder Research Computing".
  Whether compute nodes have outbound internet is not documented. Test with `curl -I https://cds.climate.copernicus.eu` inside a job. If it fails, downloads run on the transfer node.

### What "as large as possible on Trailhead" means in GPU-hours

At the caps, the ceiling is 4 H200 × 24 h × 30 d ≈ 2,900 H200-hours a month (about 1.07M SU).
You will never get that: fair share drops your priority long before. A realistic, sustainable Trailhead
budget is **one 4×H200 or 3×A100 node-day per week, about 100–300 GPU-hours a month**. This is a guess,
so measure it with `sshare -U` and `sacct` after the first sweep.
Plan the work so it fits inside 24 h jobs with checkpoint and resume.

## Getting more

| Option | Size | Needs | Verified notes |
|---|---|---|---|
| Alpine **Ascent** | 450k SU over 12 months ≈ 4,100 A100-h, or ≈ 1,200 H200-h | "Approved quickly and without review" | The form asks for "PI and role". Whether a grad student can be the PI is **not stated**, so ask rc-help. Cap is 1.8M SU per research group. Ascent jobs usually outrank Trailhead in the queue. |
| Alpine **Peak** | 450k–7.5M SU | Committee review | Only after Ascent shows results. |
| NSF ACCESS **Explore** | 400k credits | Grad student may be PI, but **the advisor must be co-PI**, plus a letter on letterhead | For unfunded thesis or dissertation work. Needs an institutional email. **Discover** (1.5M) is open to grad PIs too. |
| **NAIRR** Pilot | GPU allocations, 12 months | Grad students apply with an advisor support letter; results must be open and publishable | Still running as of 2026-09-01. Monthly cycle, deadline the 15th. |
| GEE noncommercial | Community 150, **Contributor 1,000**, Partner 100,000 EECU-h/month | Contributor **requires an active billing account**; reverify every year | Possibly not needed: AlphaEarth is on a free Source Cooperative mirror. Only Dynamic World is EE-only (see CORPUS-PLAN.md). |

Candidate faculty for PI / co-PI: Greg Tucker (Landlab, Earth surface processes), or the STAT 5010 or
CSCI 5622 instructor.

## Policy: read before putting anything on CURC

> "CU Research Computing resources may not be used for non-research purposes or for personal financial
> gain and/or commercial purposes, whether for-profit or nonprofit."

GeoSoil as an open, public-data academic project is fine. **No commercial or employer work, models,
data or credentials on Alpine**, ever. The same noncommercial rule binds the GEE free tiers and NAIRR.
NAIRR additionally requires open results.

## Next steps, in order

0. **Unblock access.** SSH with the IdentiKey password still fails (open rc-help ticket). Until it is fixed, everything runs from an
   Open OnDemand Jupyter terminal using the one-line paste in `scripts/slurm/alpine-paste.sh`.
1. **Bootstrap** (`scripts/slurm/alpine_bootstrap.sbatch`, about 1 A100-hour). It:
   - points conda at `/projects`
   - builds the environment and the mamba-ssm kernels
   - builds master.parquet
   - runs the GPU smoke test.
   It also prints whether the node can reach the internet.
2. **Bring the GEE extracts.** Copy `data/processed/gee_*.parquet` to `/projects/$USER/soil-state-transformer/data/processed/`.
   Use the transfer node once SSH works, or upload through the OnDemand file browser.
   Without them the baseline is AlphaEarth + S2 only.
3. **Baseline sweep.** `scaling_sweep.sbatch`: 4 presets × up to 4 A100-hours. It gives a size-vs-skill
   baseline on both label sources (`--source openlandmap` and `--source kssl`) at close to zero cost.
4. **Corpus build.** Follow EXECUTION-PLAN.md phases 1–5, as CPU array jobs on `acpu` plus transfer-node downloads.
5. **Ties to coursework:**
   - The STAT 5010 final project (RaCA SOC + Landlab terrain, due 12-04) can run its models here.
   - The CSCI 5622 project too, but only on its public-data parts.
6. **Allocation.** Apply for Ascent (or ACCESS Explore with the advisor as co-PI) once steps 3–4 produce
   real usage numbers. Those applications ask for example jobs and SU estimates.
