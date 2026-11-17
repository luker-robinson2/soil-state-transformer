#!/bin/bash
# Next Phase 2 run on Alpine, from the OnDemand Jupyter terminal:
#   cd /projects/$USER/soil-state-transformer && git pull && bash scripts/slurm/run_phase2_static.sh <ntfy-topic>
# Submits climate + radiometrics extraction, then join + baseline when it succeeds, and (re)starts the watcher.
set -euo pipefail
topic=${1:?usage: run_phase2_static.sh <ntfy-topic>}
cd "$(dirname "$0")/../.."
mkdir -p logs
S=$(sbatch --parsable scripts/slurm/features_static.sbatch)
B=$(sbatch --parsable --dependency=afterok:"$S" scripts/slurm/baseline_c1.sbatch)
ids=/projects/$USER/.geosoil_watch_ids
echo "$S $B $(squeue -u "$USER" -h -n geosoil-bootstrap -o %i | tr '\n' ' ')" > "$ids"
pkill -f "watch_jobs.sh $topic" || true
nohup scripts/slurm/watch_jobs.sh "$topic" "$ids" 8 > logs/watch_jobs.log 2>&1 &
echo "static=$S baseline=$B watching: $(cat "$ids")"
