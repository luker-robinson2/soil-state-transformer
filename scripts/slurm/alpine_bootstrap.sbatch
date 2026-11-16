#!/bin/bash
#SBATCH --job-name=geosoil-bootstrap
#SBATCH --account=ucb-general
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/bootstrap_%j.out
# One-time Alpine setup: conda env + mamba-ssm CUDA kernels, master.parquet, then the GPU smoke test.
set -euo pipefail
module purge
module load miniforge cuda
cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
# /home is 2 GB: keep envs and the package cache in /projects.
conda config --add envs_dirs "/projects/$USER/software/conda/envs"
conda config --add pkgs_dirs "/projects/$USER/software/conda/pkgs"
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q '^geosoil '; then
  conda env create -q -f environment-cuda.yml
fi
conda activate geosoil
python -c "import mamba_ssm" 2>/dev/null || pip install -q --no-build-isolation "causal-conv1d>=1.2" "mamba-ssm>=2.2"
[ -f data/processed/master.parquet ] || python -m geosoil.data --build
python -c "import torch, mamba_ssm; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('mamba-ssm ok')"
# Can compute nodes reach the internet? Decides whether downloads run here or on dtn.rc.colorado.edu.
curl -sS -m 10 -o /dev/null -w 'internet from compute node: HTTP %{http_code}\n' https://cds.climate.copernicus.eu || echo "internet from compute node: NO"
python -m geosoil.train --variant mamba --quick --device cuda
echo "BOOTSTRAP OK"
