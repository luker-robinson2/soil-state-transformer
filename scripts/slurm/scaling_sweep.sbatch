#!/bin/bash
#SBATCH --job-name=geosoil-scale
#SBATCH --account=ucb-general
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/scale_%A_%a.out
# Scaling study: one A100 job per (size preset, label source): size vs skill on OpenLandMap and on KSSL lab truth.
# Submit: sbatch scripts/slurm/scaling_sweep.sbatch
set -euo pipefail
module load miniforge cuda
eval "$(conda shell.bash hook)"
conda activate geosoil
cd "$SLURM_SUBMIT_DIR"
SIZES=(small base large xl)
SOURCES=(openlandmap kssl)
SIZE=${SIZES[$((SLURM_ARRAY_TASK_ID % 4))]}
SOURCE=${SOURCES[$((SLURM_ARRAY_TASK_ID / 4))]}
echo ">>> spatial-CV bake-off for size=$SIZE source=$SOURCE on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m geosoil.train --variant jepa --size "$SIZE" --source "$SOURCE" --precision bf16 --device cuda --epochs 150
echo "DONE size=$SIZE source=$SOURCE"
