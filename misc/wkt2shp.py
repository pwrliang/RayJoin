#!/usr/bin/env python3
import sys
import geopandas

in_file = sys.argv[1]
out_file = sys.argv[2]

with open(in_file, 'r') as fi:
    wkts = []
    for line in fi:
        line = line.strip()
        if len(line) > 0:
            wkts.append(line)
    df = geopandas.GeoDataFrame(geometry=geopandas.GeoSeries.from_wkt(wkts),
                                crs="EPSG:4326")
    df.to_file(out_file)
