#!/usr/bin/env bash
set -e
RELEASE_BIN_HOME="/tmp/tmp.xAcsd0C6SW/cmake-build-release-dl190"
DEBUG_BIN_HOME="/tmp/tmp.xAcsd0C6SW/cmake-build-debug-dl190"
MAPS=(Aquifers.cdb dtl_cnty.cdb Parks.cdb USACensusBlockGroupBoundaries.cdb USADetailedWaterBodies.cdb USAZIPCodeArea.cdb)
DATASET_ROOT="/local/storage/liang/rt_datasets/cdb"
DEFAULT_SEG_LEN="0.01"
DEFAULT_NE=5000000
DEFAULT_GRID_SIZE=2048
DEFAULT_XSECT_FACTOR="0.1"
DEFAULT_N_WARMUP=5
DEFAULT_N_REPEAT=5
SERIALIZE_PREFIX="/dev/shm"

function varying_grid_size() {
  debug=$1
  profile=$2
  # query_sizes=(32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864)
  seg_lens=(0.01 0.03 0.05 0.07 0.09)
  size_list=(512 1024 2048 4096 8192)
  out_dir="grid_size"
  exec="${RELEASE_BIN_HOME}"/bin/query_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${DEBUG_BIN_HOME}/bin/query_exec"
  fi

  mkdir -p "$out_dir"

  mode=grid_lb
  xsect_factor=$DEFAULT_XSECT_FACTOR
  n_warmup=$DEFAULT_N_WARMUP
  n_repeat=$DEFAULT_N_REPEAT
  if [[ $debug -eq 1 || $profile -eq 1 ]]; then
    n_warmup=1
    n_repeat=1
  fi

  if [[ $mode == "grid_lb" ]]; then
    mode="grid"
    lb=true
  else
    lb=false
  fi

  for ((i = 0; i < "${#MAPS[@]}"; i++)); do
    map="${MAPS[i]}"
    for seg_len in ${seg_lens[*]}; do
      for grid_size in ${size_list[*]}; do
        out_prefix="${out_dir}/lsi_${map}_${mode}_${grid_size}_${seg_len}"

        log_file="${out_prefix}.log"

        cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                   -serialize=${SERIALIZE_PREFIX} \
                   -grid_size=$grid_size \
                   -mode=$mode \
                   -lb=$lb \
                   -v=1 \
                   -query=lsi \
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

        if [[ $profile -eq 1 ]]; then
          profile_file="${out_prefix}"

          if [[ $mode == "grid" || $mode == "grid_lb" ]]; then
            regex="Query\|AddMapsToGrid"
          elif [[ $mode == "lbvh" ]]; then
            regex="Query\|construct\|AddMapsToGrid"
          elif [[ $mode == "rt" ]]; then
            regex="optix\|raygen"
          fi

          if [[ ! -f "${profile_file}.ncu-rep" ]]; then
            eval ncu --set full \
              --call-stack --nvtx \
              --kernel-name-base demangled \
              -k regex:"$regex" \
              -o "$profile_file" \
              "$cmd"
          fi
        fi
      done
    done
  done
}

function varying_seg_len() {
  debug=$1
  profile=$2
  seg_lens=(0.01 0.03 0.05 0.07 0.09)
  out_dir="seg_len"
  exec="${RELEASE_BIN_HOME}"/bin/query_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${DEBUG_BIN_HOME}/bin/query_exec"
  fi
  mkdir -p "$out_dir"

  n_warmup=$DEFAULT_N_WARMUP
  n_repeat=$DEFAULT_N_REPEAT
  if [[ $debug -eq 1 || $profile -eq 1 ]]; then
    n_warmup=1
    n_repeat=1
  fi

  for ((i = 0; i < "${#MAPS[@]}"; i++)); do
    map="${MAPS[i]}"

    for seg_len in ${seg_lens[*]}; do
      for mode in lbvh grid_lb; do
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

        if [[ $profile -eq 1 ]]; then
          profile_file="${out_prefix}"

          if [[ $mode == "grid" || $mode == "grid_lb" ]]; then
            regex="Query\|AddMapsToGrid"
          elif [[ $mode == "lbvh" ]]; then
            regex="Query\|construct\|AddMapsToGrid"
          elif [[ $mode == "rt" ]]; then
            regex="optix\|raygen"
          fi

          if [[ ! -f "${profile_file}.ncu-rep" ]]; then
            eval ncu --set full \
              --call-stack --nvtx \
              --kernel-name-base demangled \
              -k regex:"$regex" \
              -o "$profile_file" \
              "$cmd"
          fi
        fi
      done
    done
  done
}

function varying_query_size() {
  debug=$1
  profile=$2
  query_sizes=(5000000 7000000 9000000 11000000 13000000 15000000)
  out_dir="query_size"
  exec="${RELEASE_BIN_HOME}"/bin/query_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${DEBUG_BIN_HOME}/bin/query_exec"
  fi
  mkdir -p "$out_dir"

  xsect_factor=$DEFAULT_XSECT_FACTOR
  n_warmup=$DEFAULT_N_WARMUP
  n_repeat=$DEFAULT_N_REPEAT
  if [[ $debug -eq 1 || $profile -eq 1 ]]; then
    n_warmup=1
    n_repeat=1
  fi

  for ((i = 0; i < "${#MAPS[@]}"; i++)); do
    map="${MAPS[i]}"

    for ne in ${query_sizes[*]}; do
      for mode in lbvh grid_lb; do
        out_prefix="${out_dir}/lsi_${map}_${mode}_${ne}"

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
                   -xsect_factor $xsect_factor \
                   -gen_n=$ne \
                   -gen_t=$DEFAULT_SEG_LEN \
                   -warmup=$n_warmup \
                   -repeat=$n_repeat"

        if [[ ! -f "${log_file}" ]]; then
          echo "$cmd" > "${log_file}.tmp"
          eval "$cmd" 2>&1 | tee -a "${log_file}.tmp"

          if [[ $? -eq 0 ]]; then
            mv "${log_file}.tmp" "${log_file}"
          fi
        fi

        if [[ $profile -eq 1 ]]; then
          profile_file="${out_prefix}"

          if [[ $mode == "grid" || $mode == "grid_lb" ]]; then
            regex="Query\|AddMapsToGrid"
          elif [[ $mode == "lbvh" ]]; then
            regex="Query\|construct\|AddMapsToGrid"
          elif [[ $mode == "rt" ]]; then
            regex="optix\|raygen"
          fi

          if [[ ! -f "${profile_file}.ncu-rep" ]]; then
            eval ncu --set full \
              --call-stack --nvtx \
              --kernel-name-base demangled \
              -k regex:"$regex" \
              -o "$profile_file" \
              "$cmd"
          fi
        fi
      done
    done
  done
}

function varying_n_edges() {
  debug=$1
  profile=$2
  sample_rates=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
  out_dir="n_edges"
  exec="${RELEASE_BIN_HOME}"/bin/query_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${DEBUG_BIN_HOME}/bin/query_exec"
  fi
  mkdir -p "$out_dir"

  for ((i = 0; i < "${#MAPS[@]}"; i++)); do
    map="${MAPS[i]}"

    for sample_rate in ${sample_rates[*]}; do
      for mode in lbvh grid rt; do
        out_prefix="${out_dir}/lsi_${map}_${mode}_${sample_rate}"
        log_file="${out_prefix}.log"

        cmd="$exec -poly1 ${DATASET_ROOT}/${map} \
                   -serialize=${SERIALIZE_PREFIX} \
                   -grid_size $DEFAULT_GRID_SIZE \
                   -mode=$mode \
                   -lb=false \
                   -v=1 \
                   -query=lsi \
                   -xsect_factor $DEFAULT_XSECT_FACTOR \
                   -sample=edges \
                   -sample_rate=$sample_rate \
                   -gen_n=2 \
                   -gen_t=$DEFAULT_SEG_LEN"

        if [[ ! -f "${log_file}" ]]; then
          eval "$cmd" 2>&1 | tee "$log_file"
        fi

        if [[ $profile -eq 1 ]]; then
          profile_file="${out_prefix}"

          if [[ $mode == "grid" || $mode == "grid_lb" ]]; then
            regex="AddMapsToGrid"
          elif [[ $mode == "lbvh" ]]; then
            regex="construct\|AddMapsToGrid"
          elif [[ $mode == "rt" ]]; then
            regex="optix\|raygen"
          fi

          if [[ ! -f "${profile_file}.ncu-rep" ]]; then
            eval ncu --set full \
              --call-stack --nvtx \
              --kernel-name-base demangled \
              -k regex:"$regex" \
              -o "$profile_file" \
              "$cmd -warmup 5 -repeat 1"
          fi
        fi
      done
    done
  done
}

function construction_cost() {
  debug=$1
  profile=$2
  out_dir="construction_cost"
  exec="${RELEASE_BIN_HOME}"/bin/query_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${DEBUG_BIN_HOME}/bin/query_exec"
  fi
  mkdir -p "$out_dir"
  ne=2

  for ((i = 0; i < "${#MAPS[@]}"; i++)); do
    map="${MAPS[i]}"

    for mode in lbvh grid_lb rt; do
      out_prefix="${out_dir}/${map}_${mode}"

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
                   -grid_size $DEFAULT_GRID_SIZE \
                   -mode=$mode \
                   -lb=$lb \
                   -v=1 \
                   -query=lsi \
                   -xsect_factor $xsect_factor \
                   -gen_n=$ne \
                   -gen_t=$DEFAULT_SEG_LEN \
                   -warmup=1 \
                   -repeat=1 \
                   -profile"

      if [[ ! -f "${log_file}" ]]; then
        eval "$cmd" 2>&1 | tee "${log_file}.tmp"

        if [[ $? -eq 0 ]]; then
          mv "${log_file}.tmp" "${log_file}"
        fi
      fi
    done
  done
}

function run_all() {
  len="${#MAPS[@]}"
  NE=1000000
  mkdir -p "run_all"

  for ((i = 0; i < "$len"; i++)); do
    map="${MAPS[i]}"
    grid_res=4096
    for mode in lbvh grid grid_lb; do
      out_prefix="run_all/lsi_${map}_${mode}"

      if [[ $mode == "grid_lb" ]]; then
        mode="grid"
        lb=true
      else
        lb=false
      fi
      log_file="${out_prefix}.log"

      if [[ ! -f "${log_file}" ]]; then
        "$RELEASE_BIN_HOME"/query_exec -poly1 "${DATASET_ROOT}/${map}" \
          -serialize=${SERIALIZE_PREFIX} \
          -grid_size "$grid_res" \
          -mode="$mode" \
          -lb="$lb" \
          -v=1 \
          -gen_n=$NE \
          -gen_t=$DEFAULT_SEG_LEN 2>&1 | tee "$log_file"
      fi

      #    profile_file="lsi_${map}_${mode}_profile"
      #    kernel_name="IntersectEdge"
      #
      #    if [[ ! -f "${profile_file}.ncu-rep" ]]; then
      #      ncu --set detailed --kernel-name-base demangled -k regex:"$kernel_name" -o "$profile_file" \
      #        /tmp/tmp.xAcsd0C6SW/cmake-build-release-dl190/bin/polyover_exec -poly1 "${DATASET_ROOT}/${map1}" \
      #        -poly2 "${DATASET_ROOT}/${map2}" \
      #        -grid_size "$grid_res" \
      #        -mode="$mode" \
      #        -lb=true \
      #        -v=1
      #    fi
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
  --run-all)
    run_all $DEBUG $PROFILE
    shift
    ;;
  --seg-len)
    varying_seg_len $DEBUG $PROFILE
    shift
    ;;
  --grid-size)
    varying_grid_size $DEBUG $PROFILE
    shift
    ;;
  --query-size)
    varying_query_size $DEBUG $PROFILE
    shift
    ;;
  --num-edges)
    varying_n_edges $DEBUG $PROFILE
    shift
    ;;
  --construction-cost)
    construction_cost $DEBUG $PROFILE
    shift
    ;;
  --* | -*)
    echo "Unknown option $i"
    exit 1
    ;;
  *) ;;
  esac
done
