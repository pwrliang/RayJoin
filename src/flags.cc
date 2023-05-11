#include "flags.h"

DEFINE_string(poly1, "", "path of the first group polygons");
DEFINE_string(poly2, "", "path of the second group polygons");
DEFINE_string(output, "", "output path of overlay result");
DEFINE_int32(grid_size, 2048, "Uniform grid size");
DEFINE_double(xsect_factor, 0.1, "For preallocate intersection space");
DEFINE_string(mode, "", "grid,rt,lbvh");
DEFINE_bool(box, false, "");
DEFINE_bool(check, true, "Compare RT result with CUDA");
DEFINE_bool(lb, true, "Load balancing");
DEFINE_bool(triangle, false, "Use triangle for RT");
DEFINE_double(
    early_term_deviant, 1,
    "Stop searching when the delta of hit distance is greater the threshold");
DEFINE_double(epsilon, 0.00001, "");
DEFINE_int32(rounding_iter, 1, "Rounding iterations with nextafter");
DEFINE_bool(fau, false, "Free after use");
DEFINE_int32(warmup, 5, "");
DEFINE_int32(repeat, 5, "");
DEFINE_string(serialize, "", "");
DEFINE_bool(dynamic, false, "Dynamic query");
// Queries
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
DEFINE_string(
    profile, "",
    "Enable profiling, some counters are only compiled available under debug");