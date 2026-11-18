#!/bin/bash
# Add newly ingested label sources (e.g. SISLAC, 11-17) to the feature store without re-extracting
# everything: rebuild C1, find sites with no features yet, extract only those, refresh static sources.
#   bash scripts/slurm/run_topup.sh [after-job-id] [ntfy-ids-file]      # full: corpus rebuild first
#   SKIP_CORPUS=1 bash scripts/slurm/run_topup.sh                        # features only, labels already built
# Each top-up gets a task count no earlier shard in that directory uses, so its shards never collide.
set -euo pipefail
cd "$(dirname "$0")/../.."
AFTER=${1:+--dependency=afterany:$1}
IDS=${2:-/projects/$USER/.geosoil_watch_ids}
module load miniforge >/dev/null 2>&1 || true
set +u; eval "$(conda shell.bash hook)"; conda activate geosoil; set -u
F=data/processed/features
NA=$(python -m geosoil.features.topup --reserve $F/alphaearth 16)    # reserved at submit: concurrent top-ups can't collide
NT=$(python -m geosoil.features.topup --reserve $F/terrain 4)
NM=$(python -m geosoil.features.topup --reserve $F/terrain_ms 4)
PLAN=data/processed/features/alphaearth/plan_topup_$NA.parquet      # own plan file: a running job's plan is never overwritten

if [ -z "${SKIP_CORPUS:-}" ]; then
  C=$(sbatch --parsable $AFTER --export=ALL,SKIP_PLAN=1 scripts/slurm/corpus_build.sbatch)
  DEP="--dependency=afterok:$C"
else
  C=""; DEP="$AFTER"
fi
P=$(sbatch --parsable $DEP --job-name=geosoil-topup-plan --account=ucb-general \
      --partition=acpu --qos=cpu-normal --nodes=1 --cpus-per-task=2 --mem=16G --time=00:30:00 \
      --output=logs/topup_plan_%j.out --export=ALL,PLAN=$PLAN scripts/slurm/topup_plan.sh)
A=$(sbatch --parsable --dependency=afterok:$P --array=0-$((NA-1)) \
      --export=ALL,NTASKS=$NA,NEIGHBORS=0,YEARS=2017-2025,PLAN=$PLAN \
      scripts/slurm/features_aef.sbatch)
T=$(sbatch --parsable --dependency=afterok:$P --array=0-$((NT-1)) \
      --export=ALL,NTASKS=$NT,LABELS=data/processed/labels_new.parquet scripts/slurm/features_terrain.sbatch)
M=$(sbatch --parsable --dependency=afterok:$P --array=0-$((NM-1)) \
      --export=ALL,NTASKS=$NM,LABELS=data/processed/labels_new.parquet scripts/slurm/features_terrain_ms.sbatch)
if [ -z "${SKIP_CORPUS:-}" ]; then
  S=$(sbatch --parsable --dependency=afterok:$C --export=ALL,SOURCE=all scripts/slurm/features_static.sbatch)
else
  S=""
fi
printf "%s\n" $C $P $A $T $M $S >> "$IDS"
echo "corpus ${C:-skipped}  plan $P  alphaearth $A (ntasks $NA)  terrain $T ($NT)  terrain_ms $M ($NM)  static ${S:-skipped}"
