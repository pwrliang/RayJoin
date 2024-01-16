import numpy as np

for i in range(1, 11):
    print("awk 'BEGIN {srand()} !/^$/ { if (rand() <= .00" + str(
        i) + ") print $0}' all_nodes_lat_lon.tsv > all_nodes_lat_lon_0.00" + str(i) + ".tsv")
