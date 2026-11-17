#!/bin/bash
#SBATCH --job-name=geosoil-aef-cost
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/aef_cost_%j.out
# EXECUTION-PLAN 0.6 on Alpine: AlphaEarth read cost from Source Cooperative at 40 KSSL sites.
# Needs outbound internet from the compute node (the bootstrap job reports whether it has it).
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; set -u
set +u; conda activate geosoil; set -u   # conda activate trips over set -u
python -c "import rasterio, pyproj" 2>/dev/null || pip install -q "rasterio>=1.4" "pyproj>=3.6"
mkdir -p data/raw/aef
python scripts/corpus/aef_read_cost.py --points scripts/corpus/aef_sample_points.csv \
  --n 40 --year 2020 --out geosoil/results/aef_read_cost_alpine.json
echo "AEF COST OK"
