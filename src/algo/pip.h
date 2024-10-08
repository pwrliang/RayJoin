#ifndef RAYJOIN_ALGO_PIP
#define RAYJOIN_ALGO_PIP
#include "config.h"
#include "grid/cell.h"
#include "grid/uniform_grid.h"
#include "map/map.h"
#include "util/util.h"

namespace rayjoin {
namespace dev {

template <typename POINT_T, typename COORD_T, typename INTERNAL_COORD_T,
          typename COEFFICIENT_T>
DEV_INLINE const dev::Edge<COEFFICIENT_T>* test_against_edges_of_this_cell(
    const POINT_T& src_p, unsigned int cell_x, unsigned int cell_y,
    const UniformGrid& grid, const Scaling<COORD_T, INTERNAL_COORD_T>& scaling,
    dev::Map<INTERNAL_COORD_T, COEFFICIENT_T> base_map) {
  int base_map_id = base_map.get_id(), query_map_id = !base_map_id;
  const auto& cell = grid.get_cell(cell_x, cell_y);
  auto gsize = grid.get_grid_size();
  auto ne1 = cell.ne[0], ne = cell.ne[base_map_id];
  auto begin = cell.begin;
  const dev::Edge<COEFFICIENT_T>* best_e = nullptr;
  double best_y = std::numeric_limits<double>::infinity();
  INTERNAL_COORD_T best_y_max;

  for (index_t ie = 0; ie < ne; ie++) {
    // dst map = 0, begin + ie;
    // dst map = 1, begin + ne1 + ie
    auto eid =
        grid.get_eid(begin + base_map_id * ne1 + ie);  // mapno is either 0 or 1
    auto& e = base_map.get_edge(eid);
    auto& p1 = base_map.get_point(e.p1_idx);
    auto& p2 = base_map.get_point(e.p2_idx);
    auto x_min = min(p1.x, p2.x);
    auto x_max = max(p1.x, p2.x);
    auto x_src_p = src_p.x;
    auto y_src_p = src_p.y;
    /*
     * Is point outside x bounds of this edge?  Use simulation of simplicity:
     * shift map 1 by epsilon relative to map 0.  This also eliminates vertical
     * edges.
     */
    if (x_src_p < x_min || x_src_p > x_max ||
        x_src_p == ((query_map_id == 0) ? x_min : x_max)) {
      continue;
    }

    assert(e.b != 0);
    /*
     * Is point above edge?  If point is on edge, use simulation of simplicity.
     */
    auto xsect_y = (double) (-e.a * x_src_p - e.c) / e.b;
    auto diff_y = y_src_p - xsect_y;

    if (diff_y == 0) {
      diff_y = (query_map_id == 0 ? -e.a : e.a);
    }
    if (diff_y == 0) {
      diff_y = (query_map_id == 0 ? -e.b : e.b);
    }
#ifndef NDEBUG
    if (diff_y == 0) {
      printf("Zero length edge\n");
    }
#endif

    // current point is above the current edge
    if (diff_y > 0) {
      continue;
    }

    if (xsect_y > best_y) {
      continue;
    }

    if (xsect_y == best_y) { /* Line up from point cuts this edge and another
                              * edge at the same Y.  This should only happen at
                              * a vertex.  Roundoff error should not cause it, I
                              * hope.  If im==0, we're at the right end of both
                              * edges.  im==1 => left end.  */
                             //      printf("Tie\n");
      /* flag:  Is this slope algebraically less than the last one? */

      auto current_e_slope = (double) e.a / e.b;
      auto best_e_slope = (double) best_e->a / best_e->b;
      bool flag = current_e_slope > best_e_slope;
      /* If im==0 we want the bigger slope, if im==1, the smaller. */

      if ((query_map_id && !flag) || (flag && !query_map_id))
        continue;
    }
    best_y = xsect_y;
    best_e = &e;
    best_y_max = max(p1.y, p2.y);
  }

  /*
   * Check that the best edge intersects the ray within the current cell.  If
   * not, ignore it and act as if there were no edges in this cell crossing the
   * ray. If the hi end of the edge is in the cell then the intersection point
   * certainly is.
   */

  /* hi end of the edge is in the cell */
  if (calculate_cell(gsize, scaling, best_y_max) == cell_y) {
    return best_e;
  }
  // intersection point is above the cell
  if (calculate_cell(gsize, scaling, best_y) > cell_y) {
    best_e = nullptr;
  }

  return best_e;
}

}  // namespace dev
}  // namespace rayjoin

#endif  // RAYJOIN_ALGO_PIP
