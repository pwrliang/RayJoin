import re
import os
import numpy as np

l = ["dtl_cnty_USAZIPCodeArea_",
     "USACensusBlockGroupBoundaries_USADetailedWaterBodies_",
     "lakes_parks_Africa_",
     "lakes_parks_Asia_",
     "lakes_parks_Australia_",
     "lakes_parks_Europe_",
     "lakes_parks_North_America_",
     "lakes_parks_South_America_"]

for m in ("grid", "lbvh", "rt"):
    build_time = []
    xsect_time = []
    pip1_time = []
    pip2_time = []
    out_polygon_time = []
    for n in l:
        file = "overlay/" + n + m + ".log"
        if os.path.exists(file):
            with open(file, 'r') as fi:
                for line in fi:
                    line = line.strip()
                    x = re.search("Build Index: (.*?) ms$", line)
                    if x:
                        build_time.append(float(x.groups()[0]))
                    x = re.search("Intersection edges: (.*?) ms$", line)
                    if x:
                        xsect_time.append(float(x.groups()[0]))
                    x = re.search("Map 0: Locate vertices in other map: (.*?) ms$", line)
                    if x:
                        pip1_time.append(float(x.groups()[0]))
                    x = re.search("Map 1: Locate vertices in other map: (.*?) ms$", line)
                    if x:
                        pip2_time.append(float(x.groups()[0]))
                    x = re.search("Computer output polygons: (.*?) ms$", line)
                    if x:
                        out_polygon_time.append(float(x.groups()[0]))

        else:
            build_time.append(0)
            xsect_time.append(0)
            pip1_time.append(0)
            pip2_time.append(0)
            out_polygon_time.append(0)
    build_time = np.asarray(build_time)
    xsect_time = np.asarray(xsect_time)
    pip1_time = np.asarray(pip1_time)
    pip2_time = np.asarray(pip2_time)
    out_polygon_time = np.asarray(out_polygon_time)
    print("Mode", m)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    print("Build", build_time)
    print("LSI", xsect_time)
    print("PIP1", pip1_time)
    print("PIP2", pip2_time)
    print("Build+LSI", build_time + xsect_time)
    print("Build+PIP2", build_time + pip2_time)
    print("Overlay", build_time + xsect_time + pip1_time + pip2_time + out_polygon_time)
