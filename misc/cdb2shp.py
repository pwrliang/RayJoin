#!/usr/bin/env python3
import shapely
import geopandas
from shapely.geometry import Point, LineString, MultiLineString, Polygon
import sys

def read_cdb(path):
    list_chain_id = []
    list_first_point = []
    list_last_point = []
    list_left_polygon = []
    list_right_polygon = []
    list_geometry = []

    with open(path, 'r') as fi:
        n_point = 0
        points = []
        for line in fi:
            line = line.strip()
            arr = line.split()
            if len(arr) == 0:
                break
            if n_point == 0:
                chain_id = int(arr[0])
                n_point = int(arr[1])
                first_point = int(arr[2])
                last_point = int(arr[3])
                left_polygon = int(arr[4])
                right_polygon = int(arr[5])

                list_chain_id.append(chain_id)
                list_first_point.append(first_point)
                list_last_point.append(last_point)
                list_left_polygon.append(left_polygon)
                list_right_polygon.append(right_polygon)
            else:
                points.append((float(arr[0]), float(arr[1])))
                n_point -= 1
                if n_point == 0:
                    list_geometry.append(LineString(points))
                    points.clear()

    d = {'chain_id': list_chain_id, 'first_point': list_first_point, 'last_point': list_last_point,
         'left_polygon': list_left_polygon,
         'right_polygon': list_right_polygon,
         'geometry': list_geometry}
    return d


in_file = sys.argv[1]
out_file = sys.argv[2]

if len(sys.argv) != 3:
        exit(1)
d = read_cdb(in_file)
gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
gdf.to_file(out_file)
