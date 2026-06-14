#!/bin/bash
#SBATCH --job-name=geosoil-scale
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/scale_%A_%a.out
# Scaling study: one A100 job per size preset (size vs performance Pareto).
# Submit: sbatch scripts/slurm/scaling_sweep.sbatch
set -euo pipefail
module load anaconda cuda
conda activate geosoil
cd "$SLURM_SUBMIT_DIR"
SIZES=(small base large xl)
SIZE=${SIZES[$SLURM_ARRAY_TASK_ID]}
echo ">>> spatial-CV bake-off for size=$SIZE on $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -m geosoil.train --variant jepa --size "$SIZE" --precision bf16 --device cuda --epochs 150
echo "DONE size=$SIZE"
