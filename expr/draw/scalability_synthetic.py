import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
import os
import numpy as np
import comm_settings
import sys
import re
import pandas as pd


def get_run_time(prefix, dist, size_list):
    time_ag = []
    time_build = []
    time_query = []
    xsects = []
    np_map2_list = []

    for n in size_list:
        path = prefix + "/{dist}_{n}.log".format(dist=dist, n=n)
        np_map2 = None
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
                m = re.search(r"Intersections: (\d+)", line)
                if m is not None:
                    xsects.append(int(m.groups()[0]))
                m = re.search(r"points: (\d+)", line)
                if m is not None:
                    np_map2 = int(m.groups()[0])
        np_map2_list.append(np_map2)

    return np.asarray(xsects), np.asarray(np_map2_list), np.asarray(time_query)


patterns = ['', '\\\\', '\\\\--', '..', '..--']
light_colors = ['#6C87EA', 'lightcoral', '#FF3333', 'lemonchiffon', '#FFDF33', 'powderblue', '#33FFFF', ]
series_id = 1


def draw(prefix):
    labels = (
        r"Uniform",
        r"Gaussian",
    )
    dist = ("uniform", "gaussian")
    size_list = (1000000, 2000000, 3000000, 4000000, 5000000)
    size_str = ("1M", "2M", "3M", "4M", "5M")
    loc = [x for x in range(len(size_list))]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 5.5))

    for i in range(len(labels)):
        ax_time = axes[0][i]
        ax_th = axes[1][i]
        fig_idx = i

        label_time = "({}) {}".format(chr(ord('a') + fig_idx), labels[fig_idx] + " - Query Time")
        label_th = "({}) {}".format(chr(ord('a') + 2 + fig_idx), labels[fig_idx] + " - Throughput")

        algo = ""
        th_unit = ""
        xsects, np, time = get_run_time(prefix, dist[i], size_list)
        if 'lsi' in prefix:
            algo = "LSI"
            th = xsects / (time / 1000) / 1000 / 1000
            th_unit = 'M Intersects/s'
        else:
            algo = "PIP"
            th = np / (time / 1000) / 1000 / 1000 / 1000
            th_unit = "G Points/s"

        ax_time.plot(loc, time, marker='*', label=algo + " - Query Time", color='black')
        ax_th.plot(loc, th, marker='x', label=algo + " - Throughput", color='black')
        print("Throughput", th)
        for ax in (ax_time, ax_th):
            ax.set_xticks(loc, size_str, rotation=0)
            ax.set_xlabel(xlabel='Number of Polygons')
            ax.autoscale(tight=True)
            ax.margins(x=0.1, y=0.5)
            ylim = list(ax.get_ylim())
            ylim[0] = 0
            ylim[1] *= 1.2
            ax.set_ylim(ylim)
            ax.legend(loc='upper left', ncol=2, handletextpad=0.2, columnspacing=0.8,
                      fontsize='medium', borderaxespad=1, borderpad=0, frameon=False)
        ax_time.set_title(label_time, verticalalignment="top")
        ax_th.set_title(label_th, verticalalignment="top")

        ax_time.set_ylabel(ylabel='Query Time (ms)', labelpad=1)
        ax_th.set_ylabel(ylabel=th_unit, labelpad=1)

    fig.tight_layout()
    fig.savefig(os.path.join(prefix, '../', os.path.dirname(prefix) + '.pdf'), format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dir = os.path.dirname(sys.argv[0])
    draw(os.path.join(dir, "scal_lsi_synthetic/"))
    draw(os.path.join(dir, "scal_pip_synthetic/"))
