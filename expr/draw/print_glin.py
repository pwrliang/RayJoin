import re
import os
import numpy as np

l = ["dtl_cnty_USAZIPCodeArea.log",
     "USACensusBlockGroupBoundaries_USADetailedWaterBodies.log",
     "lakes_parks_Africa.log",
     "lakes_parks_Asia.log",
     "lakes_parks_Australia.log",
     "lakes_parks_Europe.log",
     "lakes_parks_North_America.log",
     "lakes_parks_South_America.log"]
h2d = []
preprocessing_time = []
processing_time = []
for n in l:
    file = "GLIN/LSI/" + n
    with open(file, 'r') as fi:
        t_preprocessing = 0
        t_processing = 0
        for line in fi:
            line = line.strip()
            x = re.search("Bulk load (.*?) total.*?$", line)
            if x is not None:
                t_preprocessing = float(x.groups()[0])
            x = re.search("Search time (.*?)$", line)
            if x is not None:
                t_processing = float(x.groups()[0])

        preprocessing_time.append(t_preprocessing)
        processing_time.append(t_processing)
np.set_printoptions(formatter={'float': lambda x: "{0:0.0f}".format(x)})
print("Preprocessing", np.asarray(preprocessing_time), " ms")
print("Processing", np.asarray(processing_time), " ms")
print("Total", np.asarray(preprocessing_time) + np.asarray(processing_time), " ms")
