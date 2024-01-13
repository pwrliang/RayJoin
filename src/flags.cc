#include "flags.h"

DEFINE_string(poly1, "", "path of the first group polygons");
DEFINE_string(poly2, "", "path of the second group polygons");
DEFINE_string(output, "", "output path of overlay result");
DEFINE_int32(grid_size, 2048, "Uniform grid size");
DEFINE_double(xsect_factor, 0.2, "For preallocate intersection space");
DEFINE_string(mode, "", "grid,rt,lbvh");
DEFINE_bool(box, false, "");
DEFINE_bool(check, true, "Compare RT result with CUDA");
DEFINE_bool(fau, false, "Free after use");
DEFINE_int32(warmup, 5, "");
DEFINE_int32(repeat, 5, "");
DEFINE_string(serialize, "", "");
DEFINE_int32(ag, 1, "AG method, 0: disable, 1: paper stated, 2: simple");
DEFINE_int32(ag_iter, 5, "Max iteration of grouping");
DEFINE_int32(win, 32, "Window size for AABB compression");
DEFINE_double(enlarge, 5, "Threshold for AABB merge");
// Queries
DEFINE_int32(sample_map_id, -1, "0->poly1 / 1->poly2");
DEFINE_string(sample, "",
              "Whether to sample edges or map. Sampling map keeps every chain, "
              "so it holds the topology of original map. As a result, it looks "
              "like a down scaled map");
DEFINE_string(query, "", "lsi/pip");
DEFINE_double(sample_rate, 1, "Edge sample rate of base map");
DEFINE_int32(seed, 0, "Random seed for sampling and data generating");
DEFINE_double(gen_t, 0.1, "Len of generated line segments");
DEFINE_int32(gen_n, 10000, "Number of generated edges/points");
DEFINE_bool(histo, false, "Whether to print histogram");
DEFINE_bool(
    profile, false,
    "Enable profiling, some counters are only compiled available under debug");