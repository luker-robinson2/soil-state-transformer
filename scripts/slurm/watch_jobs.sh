#!/bin/bash
# Post Slurm job state changes to an ntfy.sh topic, so a remote session can follow jobs
# without SSH. Job IDs are read from a file each minute, so more can be appended later.
#   nohup scripts/slurm/watch_jobs.sh <ntfy-topic> <ids-file> [hours] &
# Stops when every listed job has finished, or after [hours] (default 8).
topic=$1; ids=$2; hours=${3:-8}
url="https://ntfy.sh/$topic"
declare -A last
end=$((SECONDS + hours * 3600))
post() { curl -s -m 20 -d "$1" "$url" >/dev/null || true; }
post "watcher up on $(hostname)"
while [ $SECONDS -lt $end ]; do
  pending=0
  for j in $(cat "$ids" 2>/dev/null); do
    s=$(sacct -j "$j" -n -X -o State | head -1 | awk '{print $1}')
    [ -z "$s" ] && s=UNKNOWN
    if [ "$s" != "${last[$j]}" ]; then
      name=$(sacct -j "$j" -n -X -o JobName%40 | head -1 | awk '{print $1}')
      msg="$j $name $s"
      case $s in
        RUNNING) msg="$msg on $(squeue -j "$j" -h -o %N)";;
        COMPLETED|FAILED|TIMEOUT|CANCELLED*|OUT_OF_ME*|NODE_FAIL|PREEMPTED)
          log=$(ls logs/*_"$j".out 2>/dev/null | head -1)
          [ -n "$log" ] && msg="$msg | $(grep -vE '^\s*$' "$log" | tail -4 | tr '\n' ' ' | cut -c1-400)";;
      esac
      post "$msg"
      last[$j]=$s
    fi
    case $s in PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|UNKNOWN) pending=1;; esac
  done
  [ $pending = 0 ] && { post "all watched jobs finished"; exit 0; }
  sleep 60
done
post "watcher timed out after ${hours}h"
