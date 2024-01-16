import csv
import numpy as np
import matplotlib.pyplot as plt


def parse(name):
    time = {}

    with open(name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        first_row = True
        for row in spamreader:
            if first_row:
                first_row = False
                continue
            baseline = row[0]
            time[baseline] = np.asarray(row[1:]).astype(np.float32)
    return time


def get_labels():
    return [r"County $\bowtie$ Zipcode", r"Block $\bowtie$ Water", r"LKAF $\bowtie$ PKAF", r"LKAS $\bowtie$ PKAS",
            r"LKAU $\bowtie$ PKAU", r"LKEU $\bowtie$ PKEU",
            r"LKNA $\bowtie$ PKNA", r"LKSA $\bowtie$ PKSA"]


def draw_speedup_lsi():
    time = parse('speedup/lsi.csv')
    patterns = ['////', '\\\\\\', 'OO', '**', ]
    datasets = get_labels()
    speedup_grid = time["Grid"] / time["RayJoin"]
    speedup_lbvh = time["LBVH"] / time["RayJoin"]
    speedup_pssl = time["PSSL"] / time["RayJoin"]
    speedup_glin = time["GLIN"] / time["RayJoin"]
    loc = np.asarray([x for x in range(len(datasets))]).astype(np.float32)
    width = 0.15
    gap = 0.2

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 3.))
    ax.set_xticks(loc + 2 * width, datasets, rotation=15)
    ax.set_ylabel(ylabel="Speedup (log)", labelpad=1)

    ax.bar(loc, speedup_grid, label="Uniform Grid", width=width, hatch=patterns[0], color='none', edgecolor='black', )
    ax.bar(loc + gap, speedup_lbvh, label="LBVH", width=width, hatch=patterns[1], color='none', edgecolor='black', )
    ax.bar(loc + 2 * gap, speedup_pssl, label="PSSL", width=width, hatch=patterns[2], color='none', edgecolor='black', )
    ax.bar(loc + 3 * gap, speedup_glin, label="GLIN", width=width, hatch=patterns[3], color='none', edgecolor='black', )

    ax.set_yscale('log')
    ax.legend(loc='upper left', ncol=4, handletextpad=0.3,
              fontsize=11, borderaxespad=0.2, frameon=False)
    ax.autoscale(tight=True)
    ax.margins(x=0.02, y=0.3)
    fig.tight_layout(pad=0.1)
    fig.savefig('lsi_speedup.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def draw_speedup_pip():
    time = parse('speedup/pip.csv')
    patterns = ['////', '\\\\\\', 'OO', '**', ]
    datasets = get_labels()
    speedup_grid = time["Grid"] / time["RayJoin"]
    speedup_lbvh = time["LBVH"] / time["RayJoin"]
    speedup_raster = time["RasterJoin"] / time["RayJoin"]
    speedup_cuspatial = time["cuSpatial"] / time["RayJoin"]
    loc = np.asarray([x for x in range(len(datasets))]).astype(np.float32)
    width = 0.15
    gap = 0.2

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 3.))
    ax.set_xticks(loc + 2 * width, datasets, rotation=15)
    ax.set_ylabel(ylabel="Speedup (log)", labelpad=1)

    ax.bar(loc, speedup_grid, label="Uniform Grid", width=width, hatch=patterns[0], color='none', edgecolor='black', )
    ax.bar(loc + gap, speedup_lbvh, label="LBVH", width=width, hatch=patterns[1], color='none', edgecolor='black', )
    ax.bar(loc + 2 * gap, speedup_raster, label="RasterJoin", width=width, hatch=patterns[2], color='none',
           edgecolor='black', )
    ax.bar(loc + 3 * gap, speedup_cuspatial, label="cuSpatial", width=width, hatch=patterns[3], color='none',
           edgecolor='black', )
    ax.text(5.5, 0.5, "OOM", rotation=90, va='center')
    ax.text(6.3, 1.0, "Timeout", rotation=90, va='center')

    ax.set_yscale('log')
    ax.legend(loc='upper left', ncol=4, handletextpad=0.3,
              fontsize=11, borderaxespad=0.2, frameon=False)
    ax.autoscale(tight=True)
    ax.margins(x=0.02, y=0.3)
    fig.tight_layout(pad=0.1)
    fig.savefig('pip_speedup.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def draw_speedup_overlay():
    time = parse('speedup/overlay.csv')
    patterns = ['////', '\\\\\\', 'OO', '**', '--', 'XX']
    datasets = get_labels()
    speedup_PostGIS = time["PostGIS"] / time["RayJoin"]
    speedup_Kinetica = time["Kinetica"] / time["RayJoin"]
    speedup_EPUG_Overlay = time["EPUG-Overlay"] / time["RayJoin"]
    speedup_RasterIntervals = time["RasterIntervals"] / time["RayJoin"]
    speedup_grid = time["Grid"] / time["RayJoin"]
    speedup_lbvh = time["LBVH"] / time["RayJoin"]
    loc = np.asarray([x for x in range(len(datasets))]).astype(np.float32)
    width = 0.1
    gap = 0.14

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 3.))
    ax.set_xticks(loc + 3.5 * width, datasets, rotation=15)
    ax.set_ylabel(ylabel="Speedup (log)", labelpad=1)

    ax.bar(loc, speedup_PostGIS, label="PostGIS", width=width, hatch=patterns[0], color='none', edgecolor='black', )
    ax.bar(loc + gap, speedup_Kinetica, label="Kinetica", width=width, hatch=patterns[1], color='none',
           edgecolor='black', )
    ax.bar(loc + 2 * gap, speedup_EPUG_Overlay, label="EPUG-Overlay", width=width, hatch=patterns[2], color='none',
           edgecolor='black', )
    ax.bar(loc + 3 * gap, speedup_RasterIntervals, label="RasterIntervals", width=width, hatch=patterns[3],
           color='none',
           edgecolor='black', )
    ax.bar(loc + 4 * gap, speedup_grid, label="Uniform Grid", width=width, hatch=patterns[4], color='none',
           edgecolor='black', )
    ax.bar(loc + 5 * gap, speedup_lbvh, label="LBVH", width=width, hatch=patterns[5], color='none',
           edgecolor='black', )

    ax.text(5.65, 0.2, "OOM", rotation=90, va='center')

    ax.set_yscale('log')
    ax.legend(loc='upper left', ncol=3, handletextpad=0.3,
              fontsize=11, borderaxespad=0.2, frameon=False)
    ax.autoscale(tight=True)
    ax.margins(x=0.02, y=0.5)
    fig.tight_layout(pad=0.1)
    fig.savefig('overlay_speedup.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# draw_speedup_lsi()
draw_speedup_pip()
# draw_speedup_overlay()
