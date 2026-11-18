#!/bin/bash
#SBATCH --job-name=geosoil-local
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/local_eval_%j.out
# Model tiers (docs/research/LOCAL-MODELS.md): skill by block scale, by distance to training, and radius
# models vs the global one. Re-splits (adds 0.25° and 0.05° folds) and joins into its own table, so it never
# rewrites train_c1.parquet under another running job.
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
export OMP_NUM_THREADS=16
python -m geosoil.corpus.splits
python -m geosoil.features.join --out data/processed/train_c1_local.parquet
python -m geosoil.local_eval --table data/processed/train_c1_local.parquet --parts "${PARTS:-scales,distance,regional}"
echo "LOCAL_EVAL OK"
