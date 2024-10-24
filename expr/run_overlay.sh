#!/usr/bin/env bash
set -e
source env.sh

function run() {
  map1=$1
  map2=$2
  log_file=$3
  mode=$4
  check=false
  if [[ $mode == "rt" ]]; then
    check=true
  fi

  if [[ ! -f "${log_file}" ]]; then
    cmd="$exec -poly1 ${map1} \
             -poly2 ${map2} \
             -serialize=${SERIALIZE_PREFIX} \
             -grid_size=${DEFAULT_GRID_SIZE} \
             -mode=$mode \
             -v=1 \
             -fau \
             -xsect_factor $DEFAULT_XSECT_FACTOR \
             -enlarge=$DEFAULT_ENLARGE_LIM \
             -check=$check"

    echo "$cmd" >"${log_file}.tmp"
    eval "$cmd" 2>&1 | tee -a "${log_file}.tmp"

    if grep -q "Timing results" "${log_file}.tmp"; then
      mv "${log_file}.tmp" "${log_file}"
    fi
  fi
}

function run_overlay() {
  debug=$1
  out_dir="overlay"
  exec="${BIN_HOME_RELEASE}"/bin/polyover_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${BIN_HOME_DEBUG}/bin/polyover_exec"
  fi
  mkdir -p "$out_dir"

  for mode in grid rt lbvh; do
    for ((i = 0; i < "${#MAPS1[@]}"; i++)); do
      map1=${MAPS1[$i]}
      map2=${MAPS2[$i]}
      out_prefix="${out_dir}/${map1}_${map2}_${mode}"
      log_file="${out_prefix}.log"
      run "$DATASET_ROOT/point_cdb/${map1}/${map1}_Point.cdb" "$DATASET_ROOT/point_cdb/${map2}/${map2}_Point.cdb" "$log_file" "$mode"
    done

    for ((i = 0; i < "${#CONTINENTS[@]}"; i++)); do
      con=${CONTINENTS[$i]}
      out_prefix="${out_dir}/lakes_parks_${con}_${mode}"
      log_file="${out_prefix}.log"
      run "$DATASET_ROOT/point_cdb/lakes/$con/lakes_${con}_Point.cdb" "$DATASET_ROOT/point_cdb/parks/$con/parks_${con}_Point.cdb" "$log_file" "$mode"
    done
  done

}

DEBUG=0
PROFILE=0
for i in "$@"; do
  case $i in
  -b | --build)
    pushd "$BIN_HOME_DEBUG"
    make -j
    popd
    pushd "$BIN_HOME_RELEASE"
    make -j
    popd
    shift
    ;;
  -d | --debug)
    DEBUG=1
    shift
    ;;
  -p | --profile)
    PROFILE=1
    shift
    ;;
  -ov | --overlay)
    run_overlay $DEBUG
    shift
    ;;
  --* | -*)
    echo "Unknown option $i"
    exit 1
    ;;
  *) ;;
  esac
done
