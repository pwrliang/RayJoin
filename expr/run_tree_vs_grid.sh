#!/usr/bin/env bash
set -e
PROJ_HOME="$(
  cd "$(dirname "$0")/.." >/dev/null 2>&1
  pwd -P
)"

MAPS1=(USCounty.cdb USACensusBlockGroupBoundarie.cdb)
MAPS2=(USAParks.cdb USADetailedWaterBodies.cdb)
grid_sizes=(19000 10000)
DATASET_ROOT="/home/geng.161/Datasets/MapOverlay"

len="${#MAPS1[@]}"

for ((i = 0; i < $len; i++)); do
  map1="${MAPS1[i]}"
  map2="${MAPS2[i]}"
  grid_res=${grid_sizes[i]}
  for mode in lbvh grid; do
    log_file="tree_vs_gird_${map1}_${map2}_${mode}.log"

    if [[ ! -f "${log_file}" ]]; then
      /tmp/tmp.xAcsd0C6SW/cmake-build-release-dl190/bin/polyover_exec -poly1 "${DATASET_ROOT}/${map1}" \
        -poly2 "${DATASET_ROOT}/${map2}" \
        -grid_size "$grid_res" \
        -mode="$mode" \
        -lb=true \
        -v=1 2>"$log_file"
    fi

    profile_file="tree_vs_gird_${map1}_${map2}_${mode}_profile"
    kernel_name="IntersectEdge"

    if [[ ! -f "${profile_file}.ncu-rep" ]]; then
      ncu --set detailed --kernel-name-base demangled -k regex:"$kernel_name" -o "$profile_file" \
        /tmp/tmp.xAcsd0C6SW/cmake-build-release-dl190/bin/polyover_exec -poly1 "${DATASET_ROOT}/${map1}" \
        -poly2 "${DATASET_ROOT}/${map2}" \
        -grid_size "$grid_res" \
        -mode="$mode" \
        -lb=true \
        -v=1
    fi
  done
done
