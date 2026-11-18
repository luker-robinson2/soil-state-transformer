#!/bin/bash
#SBATCH --job-name=geosoil-terrain
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/terrain_%A_%a.out
# Copernicus DEM terrain features for every C1 site, one shard per array task (idempotent).
# Top-ups: NTASKS=4 LABELS=data/processed/labels_new.parquet sbatch --array=0-3 <this script>
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
python -m geosoil.features.terrain extract --task "$SLURM_ARRAY_TASK_ID" --ntasks "${NTASKS:-16}" --labels "${LABELS:-data/processed/labels_c1.parquet}" --threads "${THREADS:-8}"
