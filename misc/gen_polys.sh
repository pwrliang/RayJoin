#!/usr/bin/env bash
mkdir -p polys/wkt polys/shp

for dist in uniform gaussian; do
  for n in 200000 400000 600000 800000 1000000; do
    for seed in 1 2; do
      # generate polys about the center of USA
      out_wkt="./polys/wkt/${dist}_n_${n}_seed_${seed}.wkt"
      out_shp="./polys/shp/${dist}_n_${n}_seed_${seed}.shp"

      if [[ ! -f "$out_wkt" ]]; then
        ./generator.py distribution=$dist \
            cardinality=$n \
            dimensions=2 \
            geometry=polygon \
            polysize=0.001 \
            maxseg=10 \
            format=wkt \
            seed=$seed \
            affinematrix=50,0,-119,0,30,35 \
            affinematrix=50,0,-119,0,30,35 > $out_wkt
        fi

        if [[ ! -f "$out_shp" ]]; then
          ./wkt2shp.py "$out_wkt" "$out_shp"
        fi
    done
  done
done