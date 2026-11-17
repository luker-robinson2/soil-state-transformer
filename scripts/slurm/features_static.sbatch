#!/bin/bash
#SBATCH --job-name=geosoil-static
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/static_%j.out
# Static per-site features for every C1 site (SOURCE=all or a comma list): climate, radiometrics,
# thermal (LST ~8 GB), regolith (depth to bedrock 8.5 GB), lithology (GLiM 1.1 GB + SGMC 0.4 GB), hand.
# Downloads are cached in data/raw/, so reruns are fast.
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
python -c "import geopandas, pyogrio, zipfile_deflate64" 2>/dev/null || pip install -q "geopandas>=1.0" "pyogrio>=0.9" zipfile-deflate64
python -m geosoil.features.static --source "${SOURCE:-all}"
echo "STATIC OK"
