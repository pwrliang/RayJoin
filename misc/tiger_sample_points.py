import os
import sys
import random
import numpy as np

if len(sys.argv) != 3:
    print("Invalid args")
    exit(1)

tsv_path = sys.argv[1]
sample_rate = float(sys.argv[2])

if os.path.exists(tsv_path):
    print("Cannot open", tsv_path)

print("Reading", tsv_path, "sample rate", sample_rate)
n_points = 0
with bz2.open(tsv_path, "rt") as bz_file:
    for line in bz_file:
        line = line.rstrip('\n')
        if len(line) > 0:
            n_points += 1

n_sampled_points = n_points * sample_rate
print("Number of points", n_points, "sampled points", n_sampled_points)

data = np.genfromtxt(fname="foo.tsv", delimiter="\t", skip_header=1, filling_values=1)  # change filling_values as req'd to fill in missing values


# random.sample(range(n_points), n_sampled_points)
