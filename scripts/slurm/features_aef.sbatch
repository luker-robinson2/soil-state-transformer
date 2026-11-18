#!/bin/bash
#SBATCH --job-name=geosoil-aef
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-63%32
#SBATCH --output=logs/aef_%A_%a.out
# AlphaEarth block extraction, one shard per array task. Idempotent: finished shards are
# skipped, so re-run missing tasks after `python -m geosoil.features.alphaearth status`.
#   YEARS=2020 NEIGHBORS=64 sbatch scripts/slurm/features_aef.sbatch
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
python -m geosoil.features.alphaearth extract --task "$SLURM_ARRAY_TASK_ID" --ntasks "${NTASKS:-64}" \
  --threads "${THREADS:-16}" --neighbors "${NEIGHBORS:-64}" --years "${YEARS:-2020}" --plan "${PLAN:-data/processed/features/alphaearth/plan.parquet}"
