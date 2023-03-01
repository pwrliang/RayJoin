#ifndef RAYJOIN_GRID_CELL_H
#define RAYJOIN_GRID_CELL_H
#include "config.h"
#include "util/util.h"
namespace rayjoin {
namespace dev {

struct Cell {
  index_t begin;  // used to fill edges
  index_t ne[2];  // number of edges for map0, map1

  DEV_HOST Cell() : begin(0), ne{0, 0} {}
};

template <typename SCALING_T, typename COORD_T>
DEV_INLINE int calculate_cell(int grid_size, const SCALING_T& scaling,
                              const COORD_T& val) {
  auto internal_range = scaling.get_internal_range();
  auto internal_min = scaling.get_internal_min();
  auto cell_scale = (double) grid_size / internal_range * 0.999;
  return (val - internal_min) * cell_scale;
}
}  // namespace dev
}  // namespace rayjoin

#endif  // RAYJOIN_GRID_CELL_H
