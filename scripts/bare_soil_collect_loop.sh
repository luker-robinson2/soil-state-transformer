#!/bin/bash
# Collect finished bare-soil Earth Engine exports every 20 minutes and refresh gap-fill copies, until no task is
# READY or RUNNING. Runs on the laptop (Earth Engine auth lives there):
#   nohup scripts/bare_soil_collect_loop.sh > ~/.cache/geosoil-bare-soil-loop.log 2>&1 &
# PY defaults to the project's Python; set PY to an environment with earthengine-api + this repo's requirements.
cd "$(dirname "$0")/.."
PY=${PY:-python}
while true; do
  out=$($PY -m geosoil.features.gee_context --part bare_soil collect 2>&1 | tail -1 | tr -d '\n ')
  echo "$(date +%m-%d\ %H:%M) $out"
  if ! echo "$out" | grep -q '"part":"bare_soil"'; then          # collect failed (often expired auth): retry, never "done"
    echo "collect failed; retrying in 20 min (re-auth with: earthengine authenticate)"; sleep 1200; continue
  fi
  $PY -m geosoil.features.gee_context --part bare_soil fill >/dev/null 2>&1
  echo "$out" | grep -qE '"(READY|RUNNING)"' || { echo "bare soil complete"; break; }
  sleep 1200
done
