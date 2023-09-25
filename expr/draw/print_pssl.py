import re
import os
import numpy as np

l = ["dtl_cnty_Point.cdb_USAZIPCodeArea_Point.cdb.log",
     "USACensusBlockGroupBoundaries_Point.cdb_USADetailedWaterBodies_Point.cdb.log",
     "lakes_Africa_Point.cdb_parks_Africa_Point.cdb.log",
     "lakes_Asia_Point.cdb_parks_Asia_Point.cdb.log",
     "lakes_Australia_Point.cdb_parks_Australia_Point.cdb.log",
     "lakes_Europe_Point.cdb_parks_Europe_Point.cdb.log",
     "lakes_North_America_Point.cdb_parks_North_America_Point.cdb.log",
     "lakes_South_America_Point.cdb_parks_South_America_Point.cdb.log"]

compute_work_list_time = []
xsect_time = []
for n in l:
    file = "PSSL/" + n
    with open(file, 'r') as fi:
        for line in fi:
            line = line.strip()
            x = re.search("compute work list:                  (.*?)$", line)
            if x is not None:
                compute_work_list_time.append(float(x.groups()[0]))
            x = re.search("compute intersections and topology: (.*?)$", line)
            if x is not None:
                xsect_time.append(float(x.groups()[0]))
print((np.asarray(compute_work_list_time) + np.asarray(xsect_time)) * 1000)
