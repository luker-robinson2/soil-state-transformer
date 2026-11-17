#!/bin/bash
#SBATCH --job-name=geosoil-smoke
#SBATCH --account=ucb-general
#SBATCH --partition=aa100
#SBATCH --qos=gpu-testing
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke_%j.out
# 1-epoch GPU smoke: verifies CUDA + mamba-ssm + bf16 training path on an A100.
set -euo pipefail
module load miniforge cuda
set +u; eval "$(conda shell.bash hook)"; set -u
set +u; conda activate geosoil; set -u   # conda activate trips over set -u
cd "$SLURM_SUBMIT_DIR"
python -c "import torch, mamba_ssm; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('mamba-ssm ok')"
python -m geosoil.train --variant mamba --quick --device cuda
echo "SMOKE OK"
