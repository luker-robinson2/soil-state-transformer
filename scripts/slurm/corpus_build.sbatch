#!/bin/bash
#SBATCH --job-name=geosoil-corpus
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/corpus_build_%j.out
# Phase 1 + 2 prep on Alpine: build C1 (downloads every open source, needs R for the KSSL
# export), harmonize, split, noise floor, then the AlphaEarth site->block plan.
set -euo pipefail
module purge
module load miniforge R
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
python -m geosoil.corpus.build_c1
[ -n "${SKIP_PLAN:-}" ] || python -m geosoil.features.alphaearth plan --years "${YEARS:-2017-2025}"
echo "CORPUS OK"
