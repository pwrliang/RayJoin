#!/usr/bin/env bash
mkdir -p polys/wkt polys/shp

for dist in uniform gaussian; do
  for seed in 1 2; do
    for n in 1000000 2000000 3000000 4000000 5000000; do #3000000 4000000 5000000
      # generate polys about the center of USA
      out_wkt="./polys/wkt/${dist}_n_${n}_seed_${seed}.wkt"
      out_shp="./polys/shp/${dist}_n_${n}_seed_${seed}.shp"

      if [[ $seed -eq 1 && $n -eq 5000000 ]] || [[ $seed -eq 2 ]]; then
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
            affinematrix=50,0,-119,0,30,35 >$out_wkt
        fi

        if [[ ! -f "$out_shp" ]]; then
          ./wkt2shp.py "$out_wkt" "$out_shp" &
        fi
      fi
    done
  done
done

for job in `jobs -p`
do
echo $job
    wait $job || let "FAIL+=1"
done

echo $FAIL

if [ "$FAIL" == "0" ];
then
echo "YAY!"
else
echo "FAIL! ($FAIL)"
fi