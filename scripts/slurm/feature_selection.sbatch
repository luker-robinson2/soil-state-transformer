#!/bin/bash
#SBATCH --job-name=geosoil-featsel
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/feature_selection_%j.out
# Step 2.3: grouped permutation, SHAP share and drop-group ablation per target under spatial CV.
# Needs data/processed/train_c1.parquet (written by baseline_c1.sbatch's join step).
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
export OMP_NUM_THREADS=16
python -m geosoil.feature_selection --drop-group --max-rows "${MAX_ROWS:-150000}"
echo "FEATSEL OK"
