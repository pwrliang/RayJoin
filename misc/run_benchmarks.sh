#!/usr/bin/env bash
set -e
PROJ_HOME="$(
  cd "$(dirname "$0")/.." >/dev/null 2>&1
  pwd -P
)"

maps1=(USCounty.cdb USACensusBlockGroupBoundarie.cdb USAZIPCodeBoundaries.cdb)
maps2=(USAquifer.cdb USADetailedWaterBodies.cdb USAParks.cdb)
DATASET_ROOT="/home/geng.161/Datasets/MapOverlay"

len="${#maps1[@]}"

for ((i = 0; i < $len; i++)); do
  rt=false
#  for opt in true false; do
    #  for rt in true false; do
        for lb in true false; do
    map1="${maps1[i]}"
    map2="${maps2[i]}"
    echo "Evaluating $map1,$map2 rt: $rt lb: $lb"
    log_file="${map1}_${map2}_rt_${rt}_lb_${lb}.log"

    if [[ ! -f "${log_file}" ]]; then
      "$PROJ_HOME"/build/bin/polyover_exec -poly1 "${DATASET_ROOT}/${map1}" \
        -poly2 "${DATASET_ROOT}/${map2}" \
        -grid_size 20000 \
        -use_rt=$rt \
        -lb=$lb \
        -v=1 2>"$log_file"
    fi
  done
  #    done
  #  done
done
