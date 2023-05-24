#!/usr/bin/env bash
set -e
BIN_HOME_RELEASE="/tmp/RayJoin/cmake-build-release-dl190"
BIN_HOME_DEBUG="/tmp/RayJoin/cmake-build-debug-dl190"
MAPS=(Aquifers.cdb dtl_cnty.cdb Parks.cdb USACensusBlockGroupBoundaries.cdb USADetailedWaterBodies.cdb USAZIPCodeArea.cdb)
DATASET_ROOT="/local/storage/liang/rt_datasets/cdb"
DEFAULT_SEG_LEN="0.2"
DEFAULT_NE=1000000
DEFAULT_GRID_SIZE=2048
DEFAULT_XSECT_FACTOR="0.5"
DEFAULT_N_WARMUP=5
DEFAULT_N_REPEAT=5
SERIALIZE_PREFIX="/dev/shm"

function lsi_varying_seg_len() {
  debug=$1
  seg_lens=(0.02 0.04 0.06 0.08 0.1)
  out_dir="seg_len"
  exec="${BIN_HOME_RELEASE}"/bin/query_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${BIN_HOME_DEBUG}/bin/query_exec"
  fi
  mkdir -p "$out_dir"

  n_warmup=$DEFAULT_N_WARMUP
  n_repeat=$DEFAULT_N_REPEAT
  if [[ $debug -eq 1 ]]; then
    n_warmup=1
    n_repeat=1
  fi

  for ((i = 0; i < "${#MAPS[@]}"; i++)); do
    map="${MAPS[i]}"

    for seg_len in ${seg_lens[*]}; do
      for mode in rt grid_lb; do
        out_prefix="${out_dir}/lsi_${map}_${mode}_${seg_len}"

        if [[ $mode == "grid_lb" ]]; then
          mode="grid"
          lb=true
        else
          lb=false
        fi
        xsect_factor=$DEFAULT_XSECT_FACTOR

        log_file="${out_prefix}.log"

        cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                   -serialize=${SERIALIZE_PREFIX} \
                   -grid_size=$DEFAULT_GRID_SIZE \
                   -mode=$mode \
                   -lb=$lb \
                   -v=1 \
                   -query=lsi \
                   -seed=1 \
                   -xsect_factor $xsect_factor \
                   -gen_n=$DEFAULT_NE \
                   -gen_t=$seg_len \
                   -warmup=$n_warmup \
                   -repeat=$n_repeat"

        if [[ ! -f "${log_file}" ]]; then
          eval "$cmd" 2>&1 | tee "${log_file}.tmp"

          if [[ $? -eq 0 ]]; then
            mv "${log_file}.tmp" "${log_file}"
          fi
        fi
      done
    done
  done
}

function lsi_varying_query_size() {
  debug=$1
  query_sizes=(1000 10000 100000 1000000 10000000)
  win_sizes=(32 32 32 16 8 8)
  enlarge_lims=(8 8 8 4 2 2)
  out_dir="mb_lsi_query_size"
  exec="${BIN_HOME_RELEASE}"/bin/query_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${BIN_HOME_DEBUG}/bin/query_exec"
  fi
  mkdir -p "$out_dir"

  xsect_factor=$DEFAULT_XSECT_FACTOR
  n_warmup=$DEFAULT_N_WARMUP
  n_repeat=$DEFAULT_N_REPEAT
  if [[ $debug -eq 1 ]]; then
    n_warmup=1
    n_repeat=1
  fi

  for ((i = 0; i < "${#MAPS[@]}"; i++)); do
    map="${MAPS[i]}"

    for ((j = 0; j < "${#query_sizes[@]}"; j++)); do
      ne="${query_sizes[j]}"
      win="${win_sizes[j]}"
      enlarge="${enlarge_lims[j]}"

      for mode in rt grid_lb lbvh; do
        out_prefix="${out_dir}/${map}_${mode}_${ne}"

        if [[ $mode == "grid_lb" ]]; then
          mode="grid"
          lb=true
        else
          lb=false
        fi
        log_file="${out_prefix}.log"

        cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                   -serialize=${SERIALIZE_PREFIX} \
                   -grid_size=$DEFAULT_GRID_SIZE \
                   -mode=$mode \
                   -lb=$lb \
                   -v=1 \
                   -query=lsi \
                   -seed=1 \
                   -xsect_factor $xsect_factor \
                   -gen_n=$ne \
                   -gen_t=$DEFAULT_SEG_LEN \
                   -warmup=$n_warmup \
                   -repeat=$n_repeat \
                   -win=$win \
                   -enlarge=$enlarge"

        if [[ ! -f "${log_file}" ]]; then
          echo "$cmd" >"${log_file}.tmp"
          eval "$cmd" 2>&1 | tee -a "${log_file}.tmp"

          if grep -q "Timing results" "${log_file}.tmp"; then
            mv "${log_file}.tmp" "${log_file}"
          fi
        fi
      done
    done
  done
}

function pip_varying_query_size() {
  debug=$1
  query_sizes=(1000 10000 100000 1000000 10000000)
  win_sizes=(32 32 32 16 8 8)
  enlarge_lims=(8 8 8 4 2 2)
  out_dir="mb_pip_query_size"
  exec="${BIN_HOME_RELEASE}"/bin/query_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${BIN_HOME_DEBUG}/bin/query_exec"
  fi
  mkdir -p "$out_dir"

  xsect_factor=$DEFAULT_XSECT_FACTOR
  n_warmup=$DEFAULT_N_WARMUP
  n_repeat=$DEFAULT_N_REPEAT
  if [[ $debug -eq 1 ]]; then
    n_warmup=1
    n_repeat=1
  fi

  for ((i = 0; i < "${#MAPS[@]}"; i++)); do
    map="${MAPS[i]}"

    for ((j = 0; j < "${#query_sizes[@]}"; j++)); do
      ne="${query_sizes[j]}"
      win="${win_sizes[j]}"
      enlarge="${enlarge_lims[j]}"

      for mode in rt grid lbvh; do
        out_prefix="${out_dir}/${map}_${mode}_${ne}"
        log_file="${out_prefix}.log"

        cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                   -serialize=${SERIALIZE_PREFIX} \
                   -grid_size=$DEFAULT_GRID_SIZE \
                   -mode=$mode \
                   -v=1 \
                   -query=pip \
                   -seed=1 \
                   -gen_n=$ne \
                   -warmup=$n_warmup \
                   -repeat=$n_repeat \
                   -win=$win \
                   -enlarge=$enlarge"

        if [[ ! -f "${log_file}" ]]; then
          echo "$cmd" >"${log_file}.tmp"
          eval "$cmd" 2>&1 | tee -a "${log_file}.tmp"

          if grep -q "Timing results" "${log_file}.tmp"; then
            mv "${log_file}.tmp" "${log_file}"
          fi
        fi
      done
    done
  done
}

DEBUG=0
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
  --lsi-seg-len)
    lsi_varying_seg_len $DEBUG
    shift
    ;;
  --lsi-query-size)
    lsi_varying_query_size $DEBUG
    shift
    ;;
  --pip-query-size)
    pip_varying_query_size $DEBUG
    shift
    ;;
  --* | -*)
    echo "Unknown option $i"
    exit 1
    ;;
  *) ;;
  esac
done
