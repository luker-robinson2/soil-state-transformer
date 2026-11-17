#!/bin/bash
# Post Slurm job state changes to an ntfy.sh topic, so a remote session can follow jobs
# without SSH. Job IDs are re-read from a file each minute, so more can be appended later.
# Array jobs are summarised as counts per state (e.g. "COMPLETED 40/64 RUNNING 20 FAILED 4").
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
    states=$(sacct -j "$j" -n -X -o State | awk '{print $1}')
    n=$(echo "$states" | grep -c .)
    [ "$n" -eq 0 ] && { states=UNKNOWN; n=1; }
    summary=$(echo "$states" | sort | uniq -c | awk '{printf "%s %s ", $2, $1}')
    if [ "$summary" != "${last[$j]}" ]; then
      name=$(sacct -j "$j" -n -X -o JobName%40 | head -1 | awk '{print $1}')
      if [ "$n" -gt 1 ]; then msg="$j $name [$n tasks] $summary"; else msg="$j $name $summary"; fi
      if ! echo "$states" | grep -qE 'PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|UNKNOWN'; then
        log=$(ls -t logs/*_"$j"*.out 2>/dev/null | head -1)
        [ -n "$log" ] && msg="$msg | $(grep -vE '^\s*$' "$log" | tail -4 | tr '\n' ' ' | cut -c1-400)"
      fi
      post "$msg"
      last[$j]=$summary
    fi
    echo "$states" | grep -qE 'PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|UNKNOWN' && pending=1
  done
  [ $pending = 0 ] && { post "all watched jobs finished"; exit 0; }
  sleep 60
done
post "watcher timed out after ${hours}h"
