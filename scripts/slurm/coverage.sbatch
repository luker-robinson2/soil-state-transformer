#!/bin/bash
#SBATCH --job-name=geosoil-coverage
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/coverage_%j.out
# Area of applicability of the training data over global land (geosoil/coverage.py): where new labels help most.
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
python -m geosoil.coverage --n "${N:-30000}"
echo "COVERAGE OK"
