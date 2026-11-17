#!/bin/bash
#SBATCH --job-name=geosoil-static
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/static_%j.out
# Static-raster features for every C1 site: TerraClimate 1991-2020 normals (~0.9 GB staged) and
# radiometrics (NURE CONUS 45 MB + Radmap Australia ~6 GB staged). Downloads are cached in data/raw/.
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
python -m geosoil.features.static --source "${SOURCE:-all}"
echo "STATIC OK"
