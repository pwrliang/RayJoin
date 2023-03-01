#!/usr/bin/env bash
set -e
PROJ_HOME="$(
  cd "$(dirname "$0")/.." >/dev/null 2>&1
  pwd -P
)"

MAP1="USCounty.cdb"
MAP2="USAParks.cdb"
DATASET_ROOT="/home/geng.161/Datasets/MapOverlay"
#GRID_RESOLUTIONS=(8192 16384 32768)
GRID_RESOLUTIONS=(15000 20000 25000 30000 35000)

for grid_res in "${GRID_RESOLUTIONS[@]}"; do
  echo "Grid Resolution: $grid_res"
  log_file="introduction_grid_res_${grid_res}.log"

  if [[ ! -f "${log_file}" ]]; then
    "$PROJ_HOME"/build/bin/polyover_exec -poly1 "${DATASET_ROOT}/${MAP1}" \
      -poly2 "${DATASET_ROOT}/${MAP2}" \
      -grid_size "$grid_res" \
      -use_rt=false \
      -lb=false \
      -v=1 2>"$log_file"
  fi
done
