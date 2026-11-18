#!/bin/bash
# Add newly ingested label sources (e.g. SISLAC, 11-17) to the feature store without re-extracting
# everything: rebuild C1, find sites with no features yet, extract only those, refresh static sources.
#   bash scripts/slurm/run_topup.sh [after-job-id] [ntfy-ids-file]
# Waits for after-job-id (e.g. a running baseline, so its comparison stays on the old labels).
set -euo pipefail
cd "$(dirname "$0")/../.."
AFTER=${1:+--dependency=afterany:$1}
IDS=${2:-/projects/$USER/.geosoil_watch_ids}

C=$(sbatch --parsable $AFTER --export=ALL,SKIP_PLAN=1 scripts/slurm/corpus_build.sbatch)
P=$(sbatch --parsable --dependency=afterok:$C --job-name=geosoil-topup-plan --account=ucb-general \
      --partition=acpu --qos=cpu-normal --nodes=1 --cpus-per-task=2 --mem=16G --time=00:30:00 \
      --output=logs/topup_plan_%j.out scripts/slurm/topup_plan.sh)
A=$(sbatch --parsable --dependency=afterok:$P --array=0-15 \
      --export=ALL,NTASKS=16,NEIGHBORS=0,YEARS=2017-2025,PLAN=data/processed/features/alphaearth/plan_topup.parquet \
      scripts/slurm/features_aef.sbatch)
T=$(sbatch --parsable --dependency=afterok:$P --array=0-3 \
      --export=ALL,NTASKS=4,LABELS=data/processed/labels_new.parquet scripts/slurm/features_terrain.sbatch)
M=$(sbatch --parsable --dependency=afterok:$P --array=0-3 \
      --export=ALL,NTASKS=4,LABELS=data/processed/labels_new.parquet scripts/slurm/features_terrain_ms.sbatch)
S=$(sbatch --parsable --dependency=afterok:$C --export=ALL,SOURCE=all scripts/slurm/features_static.sbatch)
printf "%s\n" $C $P $A $T $M $S >> "$IDS"
echo "corpus $C  plan $P  alphaearth $A  terrain $T  terrain_ms $M  static $S"
