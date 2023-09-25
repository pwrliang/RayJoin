#!/usr/bin/env bash
set -e
source env.sh

function run() {
  map1=$1
  map2=$2
  enlarge=$3
  log_file=$4

  cmd="$exec -poly1 ${map1} \
             -poly2 ${map2} \
             -serialize=${SERIALIZE_PREFIX} \
             -mode=rt \
             -query=$query \
             -v=1 \
             -fau \
             -xsect_factor $DEFAULT_XSECT_FACTOR \
             -enlarge=$enlarge \
             -check=false"

  if [[ ! -f "${log_file}" ]]; then
    echo "$cmd" >"${log_file}.tmp"
    eval "$cmd" 2>&1 | tee -a "${log_file}.tmp"

    if grep -q "Timing results" "${log_file}.tmp"; then
      mv "${log_file}.tmp" "${log_file}"
    fi
  fi
}

function ag_varying_enlarge_lim() {
  debug=$1
  query=$2
  enlarge_lims=(1 2 3 4 5 6 7 8)
  out_dir="ag_${query}_varying_enlarge"
  exec="${BIN_HOME_RELEASE}"/bin/query_exec
  if [[ $debug -eq 1 ]]; then
    out_dir="${out_dir}_debug"
    exec="${BIN_HOME_DEBUG}/bin/query_exec"
  fi
  mkdir -p "$out_dir"

  for ((j = 0; j < "${#enlarge_lims[@]}"; j++)); do
    enlarge_lim="${enlarge_lims[j]}"

    for ((i = 0; i < "${#MAPS1[@]}"; i++)); do
      map1=${MAPS1[$i]}
      map2=${MAPS2[$i]}
      out_prefix="${out_dir}/${map1}_${map2}_enlarge_${enlarge_lim}"
      log_file="${out_prefix}.log"
      run "$DATASET_ROOT/point_cdb/${map1}/${map1}_Point.cdb" \
        "$DATASET_ROOT/point_cdb/${map2}/${map2}_Point.cdb" \
        "$enlarge_lim" \
        "$log_file"
    done

    for ((i = 0; i < "${#CONTINENTS[@]}"; i++)); do
      con=${CONTINENTS[$i]}
      out_prefix="${out_dir}/lakes_parks_${con}_enlarge_${enlarge_lim}"
      log_file="${out_prefix}.log"
      run "$DATASET_ROOT/point_cdb/lakes/$con/lakes_${con}_Point.cdb" \
        "$DATASET_ROOT/point_cdb/parks/$con/parks_${con}_Point.cdb" \
        "$enlarge_lim" \
        "$log_file"
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
  --lsi-vary-enlarge-lim)
    ag_varying_enlarge_lim $DEBUG "lsi"
    shift
    ;;
  --pip-vary-enlarge-lim)
    ag_varying_enlarge_lim $DEBUG "pip"
    shift
    ;;
  --* | -*)
    echo "Unknown option $i"
    exit 1
    ;;
  *) ;;
  esac
done
