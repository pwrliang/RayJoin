#!/usr/bin/env bash
set -e
source env.sh

function run() {
  map1=$1
  map2=$2
  query=$3
  log_file=$4

  if [[ ! -f "${log_file}" ]]; then
    cmd="$exec -poly1 $map1 \
             -poly2 $map2 \
             -serialize=${SERIALIZE_PREFIX} \
             -mode=rt \
             -v=1 \
             -query=$query \
             -seed=1 \
             -xsect_factor 1 \
             -warmup=$n_warmup \
             -repeat=$n_repeat \
             -ag=0 \
             -check=false"

    echo "$cmd" >"${log_file}.tmp"
    eval "$cmd" 2>&1 | tee -a "${log_file}.tmp"

    if grep -q "Timing results" "${log_file}.tmp"; then
      mv "${log_file}.tmp" "${log_file}"
    fi
  fi
}

function lsi_scalability() {
  debug=$1
  out_dir="scal_lsi_synthetic"
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

  for dist in uniform gaussian; do
    for n in 1000000 2000000 3000000 4000000 5000000; do
      out_prefix="${out_dir}/${dist}_${n}"
      log_file="${out_prefix}.log"
      run "$DATASET_ROOT/point_cdb/synthetic/${dist}_n_5000000_seed_1.cdb" \
        "$DATASET_ROOT/point_cdb/synthetic/${dist}_n_${n}_seed_2.cdb" "lsi" "$log_file"
    done
  done
}

function pip_scalability() {
  debug=$1
  out_dir="scal_pip_synthetic"
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

  for dist in uniform gaussian; do
    for n in 1000000 2000000 3000000 4000000 5000000; do
      out_prefix="${out_dir}/${dist}_${n}"
      log_file="${out_prefix}.log"
      run "$DATASET_ROOT/point_cdb/synthetic/${dist}_n_5000000_seed_1.cdb" \
        "$DATASET_ROOT/point_cdb/synthetic/${dist}_n_${n}_seed_2.cdb" "pip" "$log_file"
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
  --lsi-query)
    lsi_scalability $DEBUG
    shift
    ;;
  --pip-query)
    pip_scalability $DEBUG
    shift
    ;;
  --* | -*)
    echo "Unknown option $i"
    exit 1
    ;;
  *) ;;
  esac
done
