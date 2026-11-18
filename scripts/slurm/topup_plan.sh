#!/bin/bash
# Sites in labels_c1 with no terrain shard yet -> labels_new.parquet, and their AlphaEarth block plan.
set -euo pipefail
module purge
module load miniforge
cd "$SLURM_SUBMIT_DIR"
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
python -m geosoil.features.topup
python -m geosoil.features.alphaearth plan --years 2017-2025 --labels data/processed/labels_new.parquet \
  --out "${PLAN:-data/processed/features/alphaearth/plan_topup.parquet}"
echo "TOPUP PLAN OK"
