#!/usr/bin/env bash
set -e
RELEASE_BIN_HOME="/tmp/tmp.xAcsd0C6SW/cmake-build-release-dl190"
DEBUG_BIN_HOME="/tmp/tmp.xAcsd0C6SW/cmake-build-debug-dl190"
MAPS=(Aquifers.cdb dtl_cnty.cdb Parks.cdb USACensusBlockGroupBoundaries.cdb USAZIPCodeArea.cdb USADetailedWaterBodies.cdb)

MAPS1=(Aquifers.cdb Parks.cdb USAZIPCodeArea.cdb)
MAPS2=(dtl_cnty.cdb USACensusBlockGroupBoundaries.cdb USADetailedWaterBodies.cdb)
DATASET_ROOT="/local/storage/liang/rt_datasets/cdb"
DEFAULT_GRID_SIZE=2048
SERIALIZE_PREFIX="/dev/shm"
DEFAULT_XSECT_FACTOR="0.1"

function varying_conserve_rep() {
  debug=$1
  profile=$2
  conserve_rep_list=(0 0.0000001 0.000001 0.00001 0.0001 0.001)
  out_dir="conserve_rep"
  exec="${RELEASE_BIN_HOME}"/bin/polyover_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${DEBUG_BIN_HOME}/bin/polyover_exec"
  fi
  mkdir -p "$out_dir"

  xsect_factor=$DEFAULT_XSECT_FACTOR

  for ((i = 0; i < "${#MAPS1[@]}"; i++)); do
    map1=${MAPS1[$i]}
    map2=${MAPS2[$i]}

    for cr in ${conserve_rep_list[*]}; do
      out_prefix="${out_dir}/overlay_${map1}_${map2}_${cr}"

      log_file="${out_prefix}.log"

      cmd="$exec -poly1 ${DATASET_ROOT}/${map1} \
                         -poly2 ${DATASET_ROOT}/${map2} \
                         -serialize=${SERIALIZE_PREFIX} \
                         -grid_size=20000 \
                         -mode=rt \
                         -triangle=false \
                         -v=1 \
                         -epsilon=$cr \
                         -check=true \
                         -fau \
                         -xsect_factor $xsect_factor"

      if [[ ! -f "${log_file}" ]]; then
        echo "$cmd" >"${log_file}.tmp"
        eval "$cmd" 2>&1 | tee -a "${log_file}.tmp"

        if [[ $? -eq 0 ]]; then
          mv "${log_file}.tmp" "${log_file}"
        fi
      fi
    done
  done

}

function varying_early_term() {
  debug=$1
  profile=$2
  early_term_dev=(10 1 0.1 0.01 0.001 0.0001 0.00001)
  out_dir="early_term"
  exec="${RELEASE_BIN_HOME}"/bin/polyover_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${DEBUG_BIN_HOME}/bin/polyover_exec"
  fi
  mkdir -p "$out_dir"

  xsect_factor=$DEFAULT_XSECT_FACTOR

  for ((i = 0; i < "${#MAPS1[@]}"; i++)); do
    map1=${MAPS1[$i]}
    map2=${MAPS2[$i]}

    for dev in ${early_term_dev[*]}; do
      out_prefix="${out_dir}/overlay_${map1}_${map2}_${dev}"

      log_file="${out_prefix}.log"

      cmd="$exec -poly1 ${DATASET_ROOT}/${map1} \
                       -poly2 ${DATASET_ROOT}/${map2} \
                       -serialize=${SERIALIZE_PREFIX} \
                       -grid_size=20000 \
                       -mode=rt \
                       -triangle=false \
                       -v=1 \
                       -early_term_deviant=$dev \
                       -fau \
                       -check=true \
                       -xsect_factor $xsect_factor"

      if [[ ! -f "${log_file}" ]]; then
        echo "$cmd" >"${log_file}.tmp"
        eval "$cmd" 2>&1 | tee -a "${log_file}.tmp"

        if [[ $? -eq 0 ]]; then
          mv "${log_file}.tmp" "${log_file}"
        fi
      fi
    done
  done
}

function run_overlay() {
  debug=$1
  out_dir="overlay"
  exec="${RELEASE_BIN_HOME}"/bin/polyover_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${DEBUG_BIN_HOME}/bin/polyover_exec"
  fi
  mkdir -p "$out_dir"

  xsect_factor=$DEFAULT_XSECT_FACTOR

  for ((i = 0; i < "${#MAPS1[@]}"; i++)); do
    map1=${MAPS1[$i]}
    map2=${MAPS2[$i]}

    for mode in grid rt rt_triangle; do
      out_prefix="${out_dir}/overlay_${map1}_${map2}_${mode}"
      log_file="${out_prefix}.log"

      if [[ $mode == "grid_lb" ]]; then
        mode="grid"
        lb=true
      else
        lb=false
      fi

      if [[ $mode == "rt_triangle" ]]; then
        mode="rt"
        triangle=true
      else
        triangle=false
      fi

      cmd="$exec -poly1 ${DATASET_ROOT}/${map1} \
                       -poly2 ${DATASET_ROOT}/${map2} \
                       -serialize=${SERIALIZE_PREFIX} \
                       -grid_size=20000 \
                       -mode=$mode \
                       -triangle=$triangle \
                       -lb=$lb \
                       -v=1 \
                       -fau \
                       -early_term_deviant=1 \
                       -check=true \
                       -xsect_factor $xsect_factor"

      if [[ ! -f "${log_file}" ]]; then
        echo "$cmd" >"${log_file}.tmp"
        eval "$cmd" 2>&1 | tee -a "${log_file}.tmp"

        if [[ $? -eq 0 ]]; then
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
    pushd "$DEBUG_BIN_HOME"
    make -j
    popd
    pushd "$RELEASE_BIN_HOME"
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
  --early-term)
    varying_early_term $DEBUG
    shift
    ;;
  --conserve-rep)
    varying_conserve_rep $DEBUG
    shift
    ;;
  --* | -*)
    echo "Unknown option $i"
    exit 1
    ;;
  *) ;;
  esac
done
