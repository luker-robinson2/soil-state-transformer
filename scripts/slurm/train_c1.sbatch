#!/bin/bash
#SBATCH --job-name=geosoil-train-c1
#SBATCH --account=ucb-general
#SBATCH --partition=ah200
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_c1_%j.out
# C1 neural trainer (roadmap I2): residual MLP, same rows and spatial folds as the LightGBM baseline.
# Needs data/processed/train_c1.parquet (written by baseline_c1.sbatch's join step).
# The A100 pool is often at its group cap; the H200 3g.71gb MIG slice usually starts at once.
# Override with: sbatch --partition=aa100 --gres=gpu:a100-40gb:1 scripts/slurm/train_c1.sbatch
set -euo pipefail
module purge
module load gcc/11.2.0 miniforge cuda
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
nvidia-smi -L
python -m geosoil.train_c1 --seeds "${SEEDS:-0,1,2}"
python -m geosoil.train_c1 --seeds "${SEEDS:-0,1,2}" --neighbours
echo "TRAIN_C1 OK"
