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

time_list = []
total_time_list = []

for m in ("grid", "lbvh", "rt"):
    load_time = []
    build_time = []
    ag_time = []
    query_time = []
    for n in l:
        file = "query_lsi/" + n + m + ".log"
        if os.path.exists(file):
            with open(file, 'r') as fi:
                for line in fi:
                    line = line.strip()
                    x = re.search("Load Data: (.*?) ms", line)
                    if x:
                        load_time.append(float(x.groups()[0]))
                    x = re.search("Build Index: (.*?) ms$", line)
                    if x:
                        build_time.append(float(x.groups()[0]))
                    x = re.search("Adaptive Grouping: (.*?) ms$", line)
                    if x:
                        ag_time.append(float(x.groups()[0]))
                    x = re.search("Query: (.*?) ms$", line)
                    if x:
                        query_time.append(float(x.groups()[0]))

        else:
            load_time.append(0)
            build_time.append(0)
            ag_time.append(0)
            query_time.append(0)
    load_time = np.asarray(load_time)
    build_time = np.asarray(build_time)
    ag_time = np.asarray(ag_time)
    query_time = np.asarray(query_time)
    if m != "rt":
        ag_time = np.zeros(len(query_time))

    print("Mode", m)
    time_list.append(query_time)
    total_time_list.append(load_time + build_time + ag_time + query_time)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.0f}".format(x)})

    print("Preprocessing", load_time + ag_time + build_time)
    print("Processing", query_time)
    print("Total", load_time + ag_time + build_time + query_time)

np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
print("Process speedup", np.minimum(time_list[0], time_list[1]) / time_list[2])
print("Preprocess speedup", np.minimum(total_time_list[0], total_time_list[1]) / total_time_list[2])
