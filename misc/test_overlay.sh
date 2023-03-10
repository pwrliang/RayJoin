#!/usr/bin/env bash
set -e
PROJ_HOME="$(
  cd "$(dirname "$0")/.." >/dev/null 2>&1
  pwd -P
)"

DATASET="$PROJ_HOME/dataset"
MAPS="$DATASET/maps"

for v in debug release; do
  echo "Test build type = $v"
  # fixme: add rt triangle
  for mode in grid grid_lb rt rt_triangle; do
    echo "Mode = $mode"

    if [[ $mode == "grid_lb" ]]; then
      mode="grid"
      lb=true
    else
      lb=false
    fi

    if [[ $mode == "rt_triangle" ]]; then
      mode="rt"
      use_triangle=true
    else
      use_triangle=false
    fi

    "$PROJ_HOME"/$v/bin/polyover_exec -poly1 "$MAPS/br_county_clean_25_odyssey_final.txt" \
      -poly2 "$MAPS/br_soil_ascii_odyssey_final.txt" \
      -output "$MAPS/br_countyXbr_soil_result.txt" \
      -mode="$mode" \
      -lb="$lb" \
      -triangle="$use_triangle" \
      -early_term_deviant=100

    diff "$MAPS/br_countyXbr_soil_result.txt" "$MAPS/br_countyXbr_soil_answer.txt" >diff.txt
    n_diff=$(wc -l diff.txt | cut -d" " -f1,1)

    if [[ $n_diff -eq 0 ]]; then
      echo "Test pasted"
    else
      echo "Test failed"
      echo "Diff:"
      cat diff.txt
      exit 1
    fi
  done
done
