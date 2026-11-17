# Contributing

GeoSoil is growing from a 6k-point study into an open soil foundation model trained on CU Research
Computing (Alpine). The plan lives in [`docs/research/`](docs/research/):

| Doc | What it covers |
|---|---|
| [CURC-PLAN.md](docs/research/CURC-PLAN.md) | Cluster access, partitions, allocations, policy |
| [SCALING-PLAN.md](docs/research/SCALING-PLAN.md) | Model ladder R0–R4, compute, overfitting guards |
| [CORPUS-PLAN.md](docs/research/CORPUS-PLAN.md) | Every large soil dataset, how to get it, sampling, feature selection |
| [ROADMAP.md](docs/ROADMAP.md) | **Start here:** current state, phases, what unlocks each phase, how to resume |
| [EXECUTION-PLAN.md](docs/research/EXECUTION-PLAN.md) | Step-by-step phases 0–5 with gates |
| [PIPELINE-DESIGN.md](docs/research/PIPELINE-DESIGN.md) | How the data and feature pipeline is built |

Pick a step from EXECUTION-PLAN.md. Each step lands as one pull request.

## Ground rules

1. **Public data, open results.** No commercial, employer or customer data, models or credentials, in
   this repo or on Alpine. CURC, the Earth Engine noncommercial tier and NAIRR all forbid commercial use.
   Before every push, this must print nothing. The portfolio site under `web/` is out of scope, since its
   case studies are deliberate.
   ```bash
   git grep -IiE 'soilmetrix|ruddenklau|preprod|gserviceaccount' -- geosoil data docs/research scripts foundation-model alphaearth-soc
   ```
2. **Register the source first.** Add the dataset or model to
   [`data/corpus/sources.yaml`](data/corpus/sources.yaml) (license, access, whether rows may be
   redistributed) before any code that downloads it. Never commit rows whose license forbids
   redistribution (LUCAS, CC-BY-NC WoSIS profiles if republished, and the LUCAS slice of OSSL).
3. **Lab truth is the target.** Report skill on lab-measured labels under spatial-block CV. Model
   products (OpenLandMap, POLARIS, SoilGrids, gNATSGO) may be auxiliary pretraining targets in a clearly
   labelled ablation. They are never evaluation labels.
4. **Deduplicate across sources** (CORPUS-PLAN.md §A5) and log every dropped row.
5. **Keep the method.** Labels carry `method_id` and depth. Don't silently convert units or methods.
6. **Splits are fixed.** Use `data/corpus/splits.parquet` (1° spatial blocks plus the 2025 hold-out year)
   once it exists. Don't re-randomize.
7. **Large files stay out of git.** `*.parquet`, `*.zarr`, `*.nc` and checkpoints are gitignored. Commit
   the script that regenerates them, and document where they live on Alpine (`/projects` or
   `/scratch/alpine`).

## Running on Alpine

- Jobs go in `scripts/slurm/`, with an explicit `--account`, `--partition` and `--qos`.
- Use `module load miniforge`, with conda envs under `/projects/$USER` (`/home` is 2 GB).
- Smoke-test on `--qos=gpu-testing` (A100 MIG slice, 10% billing) before a full run.
- CPU data jobs go on `--partition=acpu --qos=cpu-normal`.

## Pull requests

- One work item per PR. Include a test, or a reproducible notebook or script, plus the numbers it produced.
- Results JSON goes in `geosoil/results/`, named by variant, size and label source.
