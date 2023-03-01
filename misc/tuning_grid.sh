for gsize in 1000 2000 4000 8000 16000; do
  echo "GS:$gsize"
  ../cmake-build-release-dl190/bin/polyover_exec -grid_size $gsize -xsect_factor 0.1 -poly1 /home/geng.161/Datasets/MapOverlay/USCounty.cdb -poly2 /home/geng.161/Datasets/MapOverlay/USAquifers.cdb
done