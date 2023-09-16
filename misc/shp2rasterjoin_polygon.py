#!/usr/bin/env python3
import shapely
import geopandas
from shapely.geometry import Point
import sys

# How to generate expected format with ArcGIS Pro?
# 1. Polygon To Line
# 2. Remove lines that have the same FID_LEFT and FID_RIGHT
# 2. Feature Vertices To Points
# 3. Export to Shapefile
in_file = sys.argv[1]
out_file = sys.argv[2]

if len(sys.argv) != 3:
    exit(1)
df = geopandas.read_file(in_file)
df = df.to_crs(epsg=4326)

min_x = None
min_y = None
max_x = None
max_y = None

with open(out_file, 'w') as fo:
    fo.write("%d\n" % len(df))

    for id, cols in df[['OBJECTID_1', 'geometry']].iterrows():
        # print(id)
        obj_id, geometry = cols
        # for interior in geometry.interiors:
        #     print(interior)
        fo.write("1\n")
        fo.write("%d\n" % len(geometry.exterior.coords))
        for p in geometry.exterior.coords:
            x, y = p
            if min_x is None:
                max_x = min_x = x
                max_y = min_y = y
            else:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
            fo.write("%.9f %.9f\n" % (y, x))

with open(out_file+".meta", 'w') as fo:
    fo.write("Input file: %s\n" % in_file)
    fo.write("Number of polygons\n")
    fo.write("Bounds, lat: %.9f - %.9f, lon: %.9f - %.9f\n" % (min_y, max_y, min_x, max_x))
