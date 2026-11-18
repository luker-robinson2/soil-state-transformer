#!/bin/bash
#SBATCH --job-name=geosoil-baseline
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/baseline_c1_%j.out
# Join labels + splits + features, then the LightGBM spatial-CV baseline on lab labels.
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
export OMP_NUM_THREADS=16
python -m geosoil.features.join
python -m geosoil.baseline_c1 ${NEIGHBOURS:+--neighbours}   # NEIGHBOURS=1 also scores neighbour features
echo "BASELINE OK"
