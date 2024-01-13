#!/usr/bin/env bash
set -e
source env.sh

DEFAULT_N_WARMUP=5
DEFAULT_N_REPEAT=5
DEFAULT_SEG_LEN="0.2"
DEFAULT_NE=1000000

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
      for mode in rt grid; do
        out_prefix="${out_dir}/lsi_${map}_${mode}_${seg_len}"

        xsect_factor=$DEFAULT_XSECT_FACTOR

        log_file="${out_prefix}.log"

        cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                   -serialize=${SERIALIZE_PREFIX} \
                   -grid_size=$DEFAULT_GRID_SIZE \
                   -mode=$mode \
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
  sample=$2
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

      for mode in rt grid lbvh; do
        out_prefix="${out_dir}/${map}_${mode}_${ne}"

        if [[ -z $sample ]]; then
          log_file="${out_prefix}.log"
          cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                     -serialize=${SERIALIZE_PREFIX} \
                     -grid_size=$DEFAULT_GRID_SIZE \
                     -mode=$mode \
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
        else
          for rate in "${SAMPLE_RATES[@]}"; do
            log_file="${out_prefix}_sample_${sample}_${rate}.log"
            cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                       -serialize=${SERIALIZE_PREFIX} \
                       -grid_size=$DEFAULT_GRID_SIZE \
                       -sample=$sample \
                       -sample_rate=$rate \
                       -mode=$mode \
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
        fi
      done
    done
  done
}

function pip_varying_query_size() {
  debug=$1
  sample=$2
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
        if [[ -z $sample ]]; then
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
        else
          for rate in "${SAMPLE_RATES[@]}"; do
            log_file="${out_prefix}_sample_${sample}_${rate}.log"
            cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                       -serialize=${SERIALIZE_PREFIX} \
                       -grid_size=$DEFAULT_GRID_SIZE \
                       -sample=$sample \
                       -sample_rate=$rate \
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
        fi
      done
    done
  done
}

function ag_varying_win_size() {
  debug=$1
  query=$2
  win_sizes=(1 2 4 8 16 32 64)
  enlarge_lim=$DEFAULT_ENLARGE_LIM
  out_dir="ag_${query}_varying_win"
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

    for ((j = 0; j < "${#win_sizes[@]}"; j++)); do
      win="${win_sizes[j]}"
      out_prefix="${out_dir}/${map}_win_${win}"
      log_file="${out_prefix}.log"

      cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                   -serialize=${SERIALIZE_PREFIX} \
                   -mode=rt \
                   -v=1 \
                   -query=$query \
                   -seed=1 \
                   -xsect_factor $xsect_factor \
                   -gen_n=$DEFAULT_NE \
                   -gen_t=$DEFAULT_SEG_LEN \
                   -warmup=$n_warmup \
                   -repeat=$n_repeat \
                   -win=$win \
                   -enlarge=$DEFAULT_ENLARGE_LIM"

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

function ag_varying_enlarge_lim() {
  debug=$1
  query=$2
  enlarge_lims=(1 2 4 8 16 32 64)
  out_dir="ag_${query}_varying_enlarge"
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

    for ((j = 0; j < "${#enlarge_lims[@]}"; j++)); do
      enlarge_lim="${enlarge_lims[j]}"
      out_prefix="${out_dir}/${map}_enlarge_$enlarge_lim"
      log_file="${out_prefix}.log"

      cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                   -serialize=${SERIALIZE_PREFIX} \
                   -grid_size=$DEFAULT_GRID_SIZE \
                   -mode=rt \
                   -v=1 \
                   -query=$query \
                   -seed=1 \
                   -xsect_factor $xsect_factor \
                   -gen_n=$DEFAULT_NE \
                   -gen_t=$DEFAULT_SEG_LEN \
                   -warmup=$n_warmup \
                   -repeat=$n_repeat \
                   -win=$DEFAULT_WIN_SIZE \
                   -enlarge=$enlarge_lim"

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

function profile_number_of_tests() {
  query=$1
  n_queries=10000000
  out_dir="profile_${query}_${n_queries}_debug"
  exec="${BIN_HOME_DEBUG}/bin/query_exec"
  mkdir -p "$out_dir"

  for ((i = 0; i < "${#MAPS[@]}"; i++)); do
    map="${MAPS[i]}"

    for mode in rt grid lbvh; do
      out_prefix="${out_dir}/${map}_$mode"
      log_file="${out_prefix}.log"

      cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                   -serialize=${SERIALIZE_PREFIX} \
                   -grid_size=$DEFAULT_GRID_SIZE \
                   -mode=$mode \
                   -v=1 \
                   -query=$query \
                   -seed=1 \
                   -xsect_factor $DEFAULT_XSECT_FACTOR \
                   -gen_n=$n_queries \
                   -gen_t=$DEFAULT_SEG_LEN \
                   -warmup=0 \
                   -repeat=1 \
                   -profile"

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
SAMPLE=""
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
  -s=* | --sample=*)
    SAMPLE="${i#*=}"
    shift
    ;;
  --lsi-seg-len)
    lsi_varying_seg_len $DEBUG
    shift
    ;;
  --lsi-query-size)
    lsi_varying_query_size $DEBUG $SAMPLE
    shift
    ;;
  --pip-query-size)
    pip_varying_query_size $DEBUG $SAMPLE
    shift
    ;;
  --lsi-vary-win-size)
    ag_varying_win_size $DEBUG "lsi"
    shift
    ;;
  --pip-vary-win-size)
    ag_varying_win_size $DEBUG "pip"
    shift
    ;;
  --lsi-vary-enlarge-lim)
    ag_varying_enlarge_lim $DEBUG "lsi"
    shift
    ;;
  --pip-vary-enlarge-lim)
    ag_varying_enlarge_lim $DEBUG "pip"
    shift
    ;;
  --lsi-ntests)
    profile_number_of_tests "lsi"
    shift
    ;;
  --pip-ntests)
    profile_number_of_tests "pip"
    shift
    ;;
  --* | -*)
    echo "Unknown option $i"
    exit 1
    ;;
  *) ;;
  esac
done
