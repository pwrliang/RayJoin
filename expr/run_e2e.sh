#!/usr/bin/env bash
set -e
source env.sh

function run_overlay() {
  debug=$1
  out_dir="overlay"
  exec="${BIN_HOME_RELEASE}"/bin/polyover_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${BIN_HOME_DEBUG}/bin/polyover_exec"
  fi
  mkdir -p "$out_dir"

  xsect_factor=$DEFAULT_XSECT_FACTOR

  for ((i = 0; i < "${#MAPS1[@]}"; i++)); do
    map1=${MAPS1[$i]}
    map2=${MAPS2[$i]}

    for mode in grid rt lbvh; do
      out_prefix="${out_dir}/${map1}_${map2}_${mode}"
      log_file="${out_prefix}.log"

      if [[ $mode == "grid_lb" ]]; then
        mode="grid"
        lb=true
      else
        lb=false
      fi

      cmd="$exec -poly1 ${DATASET_ROOT}/${map1} \
                       -poly2 ${DATASET_ROOT}/${map2} \
                       -serialize=${SERIALIZE_PREFIX} \
                       -grid_size=20000 \
                       -mode=$mode \
                       -lb=$lb \
                       -v=1 \
                       -fau \
                       -xsect_factor $xsect_factor"

      if [[ ! -f "${log_file}" ]]; then
        echo "$cmd" >"${log_file}.tmp"
        eval "$cmd" 2>&1 | tee -a "${log_file}.tmp"

        if grep -q "Timing results" "${log_file}.tmp"; then
          mv "${log_file}.tmp" "${log_file}"
        fi
      fi
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
