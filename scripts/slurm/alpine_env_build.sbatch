#!/bin/bash
#SBATCH --job-name=geosoil-env
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/env_build_%j.out
# Builds the geosoil conda env (incl. mamba-ssm CUDA kernels) on a CPU node, so the GPU
# bootstrap only has to run the smoke test when a GPU frees up. No GPU is needed to compile.
set -euo pipefail
module purge
module load miniforge cuda
cd "$SLURM_SUBMIT_DIR"
conda config --add envs_dirs "/projects/$USER/software/conda/envs"
conda config --add pkgs_dirs "/projects/$USER/software/conda/pkgs"
set +u; eval "$(conda shell.bash hook)"; set -u
if ! conda env list | grep -q '^geosoil '; then
  conda env create -q -f environment-cuda.yml
fi
set +u; conda activate geosoil; set -u   # conda activate trips over set -u
export TORCH_CUDA_ARCH_LIST="8.0;9.0" MAX_JOBS=8   # A100 + H200
python -c "import mamba_ssm" 2>/dev/null || pip install -q --no-build-isolation "causal-conv1d>=1.2" "mamba-ssm>=2.2"
python -c "import torch, mamba_ssm; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'mamba-ssm ok')"
[ -f data/processed/master.parquet ] || python -m geosoil.data --build
echo "ENV OK"
