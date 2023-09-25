import re
import os
import numpy as np

l = ["dtl_cnty",
     "USACensusBlockGroupBoundaries",
     "lakes/Africa",
     "lakes/Asia",
     "lakes/Australia",
     "lakes/Europe",
     "lakes/North_America",
     "lakes/South_America",
     ]
# Point Render Time: 83 ms
# Poly Render Time: 920 ms
render_point_time = []
render_poly_time = []
for n in l:
    file = "rasterjoin_out/" + n + "/timing"
    found = False
    with open(file, 'r') as fi:
        for line in fi:
            line = line.strip()
            x = re.search("Point Render Time: (.*?) ms$", line)
            if x is not None:
                render_point_time.append(float(x.groups()[0]))
                found = True
            x = re.search("Poly Render Time: (.*?) ms$", line)
            if x is not None:
                render_poly_time.append(float(x.groups()[0]))
    if not found:
        render_point_time.append(0)
        render_poly_time.append(0)
print(np.asarray(render_point_time) + np.asarray(render_poly_time))
