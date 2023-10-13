import os
import sys
import bz2
from osgeo import ogr
from osgeo.osr import SpatialReference

if len(sys.argv) != 3:
    print("Invalid args")
    exit(1)

bz2_path = sys.argv[1]
shape_column = int(sys.argv[2])

out_path = bz2_path + ".shp"

if os.path.exists(bz2_path):
    print("Cannot open", bz2_path)

print("Reading", bz2_path, " shape column", shape_column)

driver = ogr.GetDriverByName('Esri Shapefile')
ds = driver.CreateDataSource(out_path)

ref = SpatialReference()
ref.SetWellKnownGeogCS("WGS84")

layer = ds.CreateLayer('', ref, ogr.wkbPolygon)
# Add one attribute
layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
defn = layer.GetLayerDefn()

geom_id = 0
n_discarded = 0
with bz2.open(bz2_path, "rt") as bz_file:
    for line in bz_file:
        arr = line.rstrip('\n').split('\t')
        shape = arr[shape_column].strip('"')
        geom = ogr.CreateGeometryFromWkt(shape)
        if geom_id % 10000 == 0:
            print("Writing,", geom_id)
        geom_id += 1

        if geom is not None and geom.GetGeometryName() == 'POLYGON':
            feat = ogr.Feature(defn)
            feat.SetField('id', geom_id)
            # Make a geometry, from Shapely object
            feat.SetGeometry(geom)
            layer.CreateFeature(feat)
        else:
            n_discarded += 1

print("Discarded", n_discarded, "geometries, rate: ",
      1.0 * n_discarded / geom_id)
