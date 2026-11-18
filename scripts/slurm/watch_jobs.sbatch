#!/bin/bash
#SBATCH --job-name=geosoil-watch
#SBATCH --account=ucb-general
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=24:00:00
#SBATCH --output=logs/watch_%j.out
# The ntfy job watcher as its own 1-core job, so it outlives the OnDemand Jupyter session.
#   TOPIC=<ntfy-topic> sbatch scripts/slurm/watch_jobs.sbatch     (resubmit daily, or chain with --dependency)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec bash scripts/slurm/watch_jobs.sh "${TOPIC:?set TOPIC}" "${IDS:-/projects/$USER/.geosoil_watch_ids}" "${HOURS:-23}"
