#!/usr/bin/env python3
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Invalid arg")
    exit(1)

out_file = sys.argv[1]

min_lat = 17.674692622
max_lat = 49.385619113
min_lon = -160.236053329
max_lon = -64.566162185
n = 10000

actual_min_lat = None
actual_max_lat = None
actual_min_lon = None
actual_max_lon = None

x_arr = np.random.uniform(min_lat, max_lat, n)
y_arr = np.random.uniform(min_lon, max_lon, n)

with open(out_file, 'w') as fo:
    for i in range(n):
        x = x_arr[i]
        y = y_arr[i]
        if actual_min_lat is None:
            actual_min_lat = x
            actual_max_lat = x
            actual_min_lon = y
            actual_max_lon = y
        else:
            actual_min_lat = min(actual_min_lat, x)
            actual_max_lat = max(actual_max_lat, x)
            actual_min_lon = min(actual_min_lon, y)
            actual_max_lon = max(actual_max_lon, y)
        fo.write('%d 0 %f %f\n' % (i, x, y))

with open(out_file + ".meta", 'w') as fo:
    fo.write("Uniform distribution\n")
    fo.write("Number of points: %d\n" % n)
    fo.write("Input range, lat: %f - %f, lon: %f - %f\n" % (min_lat, max_lat, min_lon, max_lon))
    fo.write(
        "Output range, lat: %f - %f, lon: %f - %f\n" % (actual_min_lat, actual_max_lat, actual_min_lon, actual_max_lon))
