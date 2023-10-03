import matplotlib.pyplot as plt
from matplotlib.scale import LogScale
from numpy import diff
import os
import numpy as np
import comm_settings
import sys
import re
import pandas as pd


def get_time(prefix, map, mode, suffixes):
    time_ag = []
    time_build = []
    time_query = []
    ratio_compression = []
    size_BVH = []

    for suffix in suffixes:
        path = prefix + "/{map}_{mode}_{suffix}.log".format(map=map, mode=mode, suffix=suffix)
        with open(path, 'r') as fi:
            for line in fi:
                m = re.search(r"Compress ratio: (.*?) ", line)
                if m is not None:
                    ratio_compression.append(float(m.groups()[0]))
                m = re.search(r"Adaptive Grouping: (.*) ms$", line)
                if m is not None:
                    time_ag.append(float(m.groups()[0]))
                m = re.search(r"Build Index: (.*) ms$", line)
                if m is not None:
                    time_build.append(float(m.groups()[0]))
                m = re.search(r"Query: (.*?) ms$", line)
                if m is not None:
                    time_query.append(float(m.groups()[0]))
                m = re.search(r"Output Size: (.*)$", line)
                if m is not None:
                    size_BVH.append(float(m.groups()[0]))

    return (np.asarray(size_BVH), np.asarray(ratio_compression), np.asarray(time_ag), np.asarray(time_build),
            np.asarray(time_query))


patterns = ['', '\\\\', '\\\\--', '..', '..--']
light_colors = ['#6C87EA', 'lightcoral', '#FF3333', 'lemonchiffon', '#FFDF33', 'powderblue', '#33FFFF', ]
series_id = 1


def scale_size(size_list, k_scale=1024):
    return tuple(str(int(kb)) + "K" if kb < k_scale else str(int(kb / k_scale)) + "M" for kb in
                 np.asarray(size_list) / k_scale)


width = 0.5


def draw_enlarge_lim(prefix):
    labels = (r"USCounty $\bowtie$ Zipcode", r"BlockGroup $\bowtie$ Waterbodies",
              r"LKAF $\bowtie$ PKAF",
              r"LKAS $\bowtie$ PKAS",
              r"LKAU $\bowtie$ PKAU",
              r"LKEU $\bowtie$ PKEU",
              r"LKNA $\bowtie$ PKNA",
              r"LKSA $\bowtie$ PKSA",
              )
    maps = ["dtl_cnty_USAZIPCodeArea",
            "USACensusBlockGroupBoundaries_USADetailedWaterBodies",
            "lakes_parks_Africa",
            "lakes_parks_Asia",
            "lakes_parks_Australia",
            "lakes_parks_Europe",
            "lakes_parks_North_America",
            "lakes_parks_South_America"]
    fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(36, 6))

    for i in range(16):
        row = i // 8
        col = i % 8
        ax = axes[row][col]
        fig_idx = i
        map = maps[col]

        enlarge_list = ("1", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0")

        loc = [x for x in range(len(enlarge_list))]
        size_bvh, ratio_compression, time_ag_rt, time_build_rt, time_lsi_rt = get_time(
            prefix + "/ag_lsi_varying_enlarge", map,
            "enlarge",
            enlarge_list)
        _, _, _, _, time_pip_rt = get_time(prefix + "/ag_pip_varying_enlarge", map, "enlarge", enlarge_list)
        label = "({}) {}".format(chr(ord('a') + fig_idx), labels[col])

        if row == 0:
            label += " - Exec. Time"
            ylabel = 'Execution Time (ms)'

            df = pd.DataFrame({"AG": time_ag_rt,
                               "Build BVH": time_build_rt,
                               "LSI": time_lsi_rt,
                               "PIP": time_pip_rt})

            df.plot(kind="bar", stacked=True, width=width, ax=ax, hatch='')

        else:
            label += " - Comp. Rate"
            ylabel = 'Compression Rate - $r$'
            print(ratio_compression)
            ax.plot(loc, ratio_compression, label="Comp. Rate")

        ax.set_xticks(loc, enlarge_list, rotation=0)
        ax.set_title(label, verticalalignment="top")
        ax.set_xlabel(xlabel='Merging Threshold - $s$')
        ax.set_ylabel(ylabel=ylabel, labelpad=1)
        ax.autoscale(tight=True)

        if row == 0:
            ax.margins(x=0.05, y=0.8)
        else:
            ax.set_ylim((0, 1))
            ax.margins(x=0.05, )

        ax.legend(loc='upper left', ncol=2, handletextpad=0.3,
                  fontsize='medium', borderaxespad=1, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(prefix, '../', prefix + 'ag_varying_enlarge_all.pdf'), format='pdf',
                bbox_inches='tight')
    plt.show()



def draw_enlarge_lim_pick(prefix):
    maps = ["lakes_parks_North_America", ]
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10.7, 3.5))

    map = maps[0]
    ax_total = axes[0]
    ax_bvh = axes[1]
    ax_query = axes[2]

    enlarge_list = ("1", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0")

    loc = [x for x in range(len(enlarge_list))]
    size_bvh, ratio_compression, time_ag_rt, time_build_rt, time_lsi_rt = get_time(prefix + "/ag_lsi_varying_enlarge",
                                                                                   map,
                                                                                   "enlarge",
                                                                                   enlarge_list)
    _, _, _, _, time_pip_rt = get_time(prefix + "/ag_pip_varying_enlarge", map, "enlarge", enlarge_list)
    titles = ("(a) Total Exec. Time Breakdown", "(b) BVH Construction Time", "(c) Query Time")
    y_labels = ('Execution Time (ms)', 'Execution Time (ms)', 'Execution Time (ms)')
    total_time = time_ag_rt + time_build_rt + time_lsi_rt + time_pip_rt

    print("Total", total_time)
    print("No AG", total_time[0], "Min", min(total_time), "Speedup", total_time[0] / min(total_time), "Reduce",
          (total_time[0] - min(total_time)) / total_time[0])

    df = pd.DataFrame({"AG Overhead": time_ag_rt,
                       "BVH Buildup": time_build_rt,
                       "LSI Query": time_lsi_rt,
                       "PIP Query": time_pip_rt})

    df.plot(kind="bar", stacked=True, width=width, ax=ax_total, hatch='')
    ax_bvh.plot(loc, time_build_rt, label="BVH Buildup")
    ax_query.plot(loc, time_lsi_rt, label="LSI Query")
    ax_query.plot(loc, time_pip_rt, label="PIP Query")

    dydx = diff(ratio_compression) / diff([float(x) for x in enlarge_list])

    print("dydx", dydx)

    for i in range(3):
        ax = axes[i]
        title = titles[i]
        ylabel = y_labels[i]
        ax.set_xticks(loc, enlarge_list, rotation=0)
        ax.set_title(title, verticalalignment="top")
        ax.set_xlabel(xlabel='Merging Threshold - $s$')
        ax.set_ylabel(ylabel=ylabel, labelpad=1)
        ax.autoscale(tight=True)

        ax.margins(x=0.05, y=0.25)
        # ax.set_ylim((0, 1))
        ax.legend(loc='upper left', ncol=2, handletextpad=0.3,
                  fontsize='medium', borderaxespad=0.2, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(prefix, 'ag_varying_enlarge.pdf'), format='pdf',
                bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    dir = os.path.dirname(sys.argv[0])
    # draw_enlarge_lim(dir)
    draw_enlarge_lim_pick(dir)