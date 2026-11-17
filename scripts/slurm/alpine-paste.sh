cd /projects/$USER && { [ -d soil-state-transformer ] || git clone -q https://github.com/luker-robinson2/soil-state-transformer.git; } && cd soil-state-transformer && mkdir -p logs && cat > scripts/slurm/alpine_bootstrap.sbatch <<'SBATCH'
#!/bin/bash
#SBATCH --job-name=geosoil-bootstrap
#SBATCH --account=ucb-general
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/bootstrap_%j.out
# One-time Alpine setup: conda env + mamba-ssm CUDA kernels, master.parquet, then the GPU smoke test.
set -euo pipefail
module purge
module load gcc/11.2.0 miniforge cuda   # system gcc 8.5 cannot compile torch 2.4 headers
cd "$SLURM_SUBMIT_DIR"
nvidia-smi -L
# /home is 2 GB: keep envs and the package cache in /projects.
conda config --add envs_dirs "/projects/$USER/software/conda/envs"
conda config --add pkgs_dirs "/projects/$USER/software/conda/pkgs"
set +u; eval "$(conda shell.bash hook)"; set -u
if ! conda env list | grep -q '^geosoil '; then
  conda env create -q -f environment-cuda.yml
fi
set +u; conda activate geosoil; set -u   # conda activate trips over set -u
python -c "import mamba_ssm" 2>/dev/null || pip install -q --no-build-isolation "causal-conv1d==1.4.0" "mamba-ssm==2.2.2" "transformers==4.44.2"   # prebuilt torch 2.4 wheels; mamba 2.2.2 needs pre-4.46 transformers
[ -f data/processed/master.parquet ] || python -m geosoil.data --build
python -c "import torch, mamba_ssm; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('mamba-ssm ok')"
# Can compute nodes reach the internet? Decides whether downloads run here or on dtn.rc.colorado.edu.
curl -sS -m 10 -o /dev/null -w 'internet from compute node: HTTP %{http_code}\n' https://cds.climate.copernicus.eu || echo "internet from compute node: NO"
python -m geosoil.train --variant mamba --quick --device cuda
echo "BOOTSTRAP OK"
SBATCH
sbatch scripts/slurm/alpine_bootstrap.sbatch

# --- AlphaEarth read cost on Alpine (EXECUTION-PLAN 0.6); run after the bootstrap job succeeded ---
cd /projects/$USER/soil-state-transformer && git fetch -q origin corpus-c1 && git checkout -q corpus-c1 && mkdir -p logs && sbatch scripts/slurm/aef_read_cost.sbatch
