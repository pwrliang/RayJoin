import geopandas
from shapely.geometry import Point, LineString, MultiLineString, Polygon


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


# d = read_cdb('/Users/liang/CLionProjects/epug-overlay/br_soil_ascii_odyssey_final.txt')
d = read_cdb('/Users/liang/CLionProjects/epug-overlay/br_county_clean_25_odyssey_final.txt')
gdf = geopandas.GeoDataFrame(d, crs="EPSG:4326")
gdf.to_file('br_county/br_county_clean_25_odyssey_final.shp')
# fig, ax = plt.subplots(1, 1)
# gdf.plot(ax=ax)
# fig.show()
# geoplot.polyplot(gdf, figsize=(8, 4))
