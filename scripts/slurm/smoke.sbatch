#!/bin/bash
#SBATCH --job-name=geosoil-smoke
#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke_%j.out
# 1-epoch GPU smoke: verifies CUDA + mamba-ssm + bf16 training path on an A100.
set -euo pipefail
module load anaconda cuda
conda activate geosoil
cd "$SLURM_SUBMIT_DIR"
python -c "import torch, mamba_ssm; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('mamba-ssm ok')"
PYTORCH_ENABLE_MPS_FALLBACK=1 python -m geosoil.train --variant mamba --quick --device cuda
echo "SMOKE OK"
