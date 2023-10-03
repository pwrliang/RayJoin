#!/usr/bin/env python3
import geopandas
import sys
import struct

in_file = sys.argv[1]
out_prefix = sys.argv[2]

if len(sys.argv) != 3:
    exit(1)
df = geopandas.read_file(in_file)
df = df.to_crs(epsg=4326)
offset = 0

with open(out_prefix + "_fixed_binary.dat", 'wb') as fo_binary:
    with open(out_prefix + "_offset_map.dat", 'wb') as fo_offset:
        n_polys = 0
        for original_id, row in df.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Polygon':
                n_polys += 1

        fo_binary.write(struct.pack("i", n_polys))  # total polygon count
        offset += 4

        fo_offset.write(struct.pack("i", n_polys))

        for original_id, row in df.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Polygon':
                coords = row.geometry.exterior.coords
                n_points = len(coords)
                fo_offset.write(struct.pack("i", original_id))
                fo_offset.write(struct.pack("q", offset))

                fo_binary.write(struct.pack("i", original_id))  # polygon0 ID
                offset += 4

                fo_binary.write(struct.pack("i", n_points))  # vertex count
                offset += 4
                for p in coords:
                    x, y = p
                    fo_binary.write(struct.pack("d", x))  # x
                    offset += 8
                    fo_binary.write(struct.pack("d", y))  # y
                    offset += 8
