#!/bin/bash
#SBATCH --job-name=geosoil-zones
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/zones_%j.out
# Edaphic zones (model tier T2): per-zone and zone-feature models vs the global one, and leave-one-zone-out
# transfer. Uses the train_c1.parquet written by baseline_c1.sbatch's join step.
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
export OMP_NUM_THREADS=16
python -m geosoil.zones --k "${K:-8,16,32}" --targets "${TARGETS:-soc,ph,sand,clay,bd}"
echo "ZONES OK"
