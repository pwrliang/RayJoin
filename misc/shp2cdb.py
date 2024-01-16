#!/usr/bin/env python3
import geopandas
from shapely.geometry import LineString
import sys

# How to generate expected format with ArcGIS Pro?
# 1. Polygon To Line
# 2. Export to Shapefile
in_file = sys.argv[1]
out_file = sys.argv[2]

if len(sys.argv) != 3:
    exit(1)
df = geopandas.read_file(in_file)
df = df.to_crs(epsg=4326)
n_points = 0

with open(out_file, 'w') as fo:
    for row_index, row in df.iterrows():
        left_fid = int(row['LEFT_FID'])
        right_fid = int(row['RIGHT_FID'])
        geom = row['geometry']
        assert isinstance(geom, LineString)
        coords = geom.coords
        fo.write("{obj_id} {np} {fpid} {lpid} {lid} {rid}\n".format(
            obj_id=row_index,
            np=len(coords),
            fpid=n_points,
            lpid=n_points + len(coords) - 1,
            lid=left_fid,
            rid=right_fid
        ))
        n_points += len(coords)
        for x, y in geom.coords:
            fo.write("%.9f %.9f\n" % (x, y))
