#!/bin/bash
#SBATCH --job-name=geosoil-terrain-ms
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-31
#SBATCH --output=logs/terrain_ms_%A_%a.out
# Multi-scale terrain (0.5-20 km) from Copernicus GLO-90, one contiguous block of 1-degree tiles per task.
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
python -m geosoil.features.terrain_ms extract --task "$SLURM_ARRAY_TASK_ID" --ntasks 32
