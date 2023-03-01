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
df_gb_orig_fid = df.groupby('ORIG_FID')

with open(out_file, 'w') as fo:
    point_id = 0
    for original_id, df_group in df_gb_orig_fid:
        n_points = len(df_group)
        lr_fid = None
        first_point_id = None
        last_point_id = None
        points = []
        last_p = None
        for row_index, row in df_group.iterrows():
            if lr_fid is None:
                left_fid = int(row['LEFT_FID'])
                right_fid = int(row['RIGHT_FID'])
                left_fid += 1
                right_fid += 1
                lr_fid = (left_fid, right_fid)
                first_point_id = point_id
            point_id += 1
            p = row['geometry']
            assert isinstance(p, shapely.geometry.point.Point)
            if str(last_p) != str(p):
                points.append(p)
            else:
                print("Duplicated point idx: {} {}".format(row_index, p))
            p = last_p
        last_point_id = point_id - 1
        fo.write("{obj_id} {np} {fpid} {lpid} {lid} {rid}\n".format(
            obj_id=original_id,
            np=len(points),
            fpid=first_point_id,
            lpid=last_point_id,
            lid=lr_fid[0],
            rid=lr_fid[1]
        ))
        for p in points:
            fo.write("%.9f %.9f\n" % (p.x, p.y))
