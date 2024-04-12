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
    file = "mri/rasterintervals_out/" + n
    with open(file, 'r') as fi:
        raster_gen_time = 0
        for line in fi:
            line = line.strip()
            x = re.search("Computed intervals for dataset .*? in (.*?) seconds$", line)
            if x is not None:
                raster_gen_time += float(x.groups()[0])
            x = re.search("Finished in (.*?) seconds.$", line)
            if x is not None:
                processing_time.append(float(x.groups()[0]))
        preprocessing_time.append(raster_gen_time)
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
print("Preprocessing", np.asarray(preprocessing_time), " s")
print("Processing", np.asarray(processing_time), " s")
