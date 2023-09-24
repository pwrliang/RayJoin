#ifndef RAYJOIN_FLAGS_H
#define RAYJOIN_FLAGS_H

#include <gflags/gflags.h>

#include <limits>

#include "flags.h"

DECLARE_string(poly1);
DECLARE_string(poly2);
DECLARE_string(output);
DECLARE_int32(grid_size);
DECLARE_double(xsect_factor);
DECLARE_string(mode);
DECLARE_bool(box);
DECLARE_bool(check);
DECLARE_bool(lb);
DECLARE_bool(fau);
DECLARE_int32(warmup);
DECLARE_int32(repeat);
DECLARE_string(serialize);

DECLARE_int32(win);
DECLARE_bool(new_compress);
DECLARE_int32(compress_iter);
DECLARE_double(enlarge);
// Queries
DECLARE_int32(sample_map_id);
DECLARE_string(query);
DECLARE_string(sample);
DECLARE_double(sample_rate);
DECLARE_int32(seed);
DECLARE_double(gen_t);
DECLARE_int32(gen_n);
DECLARE_bool(histo);
DECLARE_bool(profile);
#endif  // RAYJOIN_FLAGS_H
