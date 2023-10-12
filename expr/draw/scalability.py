import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
import os
import numpy as np
import comm_settings
import sys
import re
import pandas as pd


def get_lsi_time(prefix, map_names, sample_method, sample_rates):
    time_ag = []
    time_build = []
    time_query = []

    for sample in sample_rates:
        path = prefix + "/{map_names}_{method}_{sample}.log".format(map_names=map_names, method=sample_method,
                                                                    sample=sample)
        with open(path, 'r') as fi:
            for line in fi:
                m = re.search(r"Adaptive Grouping: (.*) ms$", line)
                if m is not None:
                    time_ag.append(float(m.groups()[0]))
                m = re.search(r"Build Index: (.*) ms$", line)
                if m is not None:
                    time_build.append(float(m.groups()[0]))
                m = re.search(r"Query: (.*?) ms$", line)
                if m is not None:
                    time_query.append(float(m.groups()[0]))
    return np.asarray(time_query)


patterns = ['', '\\\\', '\\\\--', '..', '..--']
light_colors = ['#6C87EA', 'lightcoral', '#FF3333', 'lemonchiffon', '#FFDF33', 'powderblue', '#33FFFF', ]
series_id = 1


def draw(prefix):
    labels = (
        r"Block $\bowtie$ Water",
        r"LKAS $\bowtie$ PKAS",
        r"LKEU $\bowtie$ PKEU",
        r"LKNA $\bowtie$ PKNA",
    )
    maps1 = ("USACensusBlockGroupBoundaries", "lakes")
    maps2 = ("USADetailedWaterBodies", "parks")
    continents = ("Africa", "Asia", "Australia", "Europe", "North_America", "South_America")
    continents = ("Asia", "Europe", "North_America",)

    sample_rates = ("0.2", "0.4", "0.6", "0.8", "1.0")
    # sample_rates = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0")
    loc = [x for x in range(len(sample_rates))]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 5.5))

    for i in range(len(labels)):
        ax = axes[i // 2][i % 2]
        fig_idx = i

        label = "({}) {}".format(chr(ord('a') + fig_idx), labels[fig_idx])
        if i == 0:
            map1 = maps1[0]
            map2 = maps2[0]
            map_names = map1 + "_" + map2
        else:
            map1 = maps1[1]
            map2 = maps2[1]
            map_names = map1 + "_" + map2 + "_" + continents[i - 1]
        algo=""
        if 'lsi' in prefix:
            marker = '*'
            method = "edges"
            algo="LSI"
        else:
            marker = 'x'
            method = "points"
            algo="PIP"
        time = get_lsi_time(prefix, map_names, method, sample_rates)
        print(map_names, time)
        ax.plot(loc, time, marker=marker, label="RayJoin-" + algo, color='black')

        ax.set_xticks(loc, sample_rates, rotation=0)
        ax.set_title(label, verticalalignment="top")
        ax.set_xlabel(xlabel='Sample Rate')
        ax.set_ylabel(ylabel='Query Time (ms)', labelpad=1)
        ax.autoscale(tight=True)
        ax.margins(x=0.1, y=0.5)
        # ylim = list(ax.get_ylim())
        # ylim[0] = 0
        # ylim[1] *= 1.5
        # ax.set_ylim(ylim)
        ax.legend(loc='upper left', ncol=2, handletextpad=0.2, columnspacing=0.8,
                  fontsize='medium', borderaxespad=1, borderpad=0, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(prefix, '../', os.path.dirname(prefix) + '.pdf'), format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dir = os.path.dirname(sys.argv[0])
    draw(os.path.join(dir, "scal_lsi/"))
    draw(os.path.join(dir, "scal_pip/"))
