
#ifndef APP_PIP_GRID_H
#define APP_PIP_GRID_H
#include "algo/pip.h"
#include "app/pip.h"
#include "grid/uniform_grid.h"

namespace rayjoin {
template <typename CONTEXT_T>
class PIPGrid : public PIP<CONTEXT_T> {
  using grid_t = UniformGrid;  // only contains 1 map
  using map_t = typename CONTEXT_T::map_t;
  using point_t = typename map_t::point_t;

 public:
  PIPGrid(CONTEXT_T& ctx, std::shared_ptr<grid_t> grid)
      : PIP<CONTEXT_T>(ctx), grid_(std::move(grid)) {}

  void Query(Stream& stream, int base_map_id,
             ArrayView<point_t> d_query_points) {
    auto& scaling = this->ctx_.get_scaling();
    auto d_grid = grid_->DeviceObject();
    auto d_base_map = this->ctx_.get_map(base_map_id)->DeviceObject();
    auto gsize = grid_->get_grid_size();
    auto n_points = d_query_points.size();

    this->closest_eids_.resize(n_points);

    ArrayView<index_t> d_closest_eids(this->closest_eids_);

    ForEach(stream, n_points, [=] __device__(size_t point_idx) mutable {
      const auto& p = d_query_points[point_idx];
      auto cx = dev::calculate_cell(gsize, scaling, p.x);
      auto cy = dev::calculate_cell(gsize, scaling, p.y);

      assert(cx >= 0 && cx < gsize);
      assert(cy >= 0 && cy < gsize);
      // init value means point is not in hit
      auto closest_eid = static_cast<index_t>(DONTKNOW);

      // shoot a ray toward positive y-axis
      for (auto curr_cy = cy; curr_cy < gsize; curr_cy++) {
        // find a closest edge of dst-map to this vertex in
        // this cell
        auto* beste = dev::test_against_edges_of_this_cell(
            p,         /* IN: current vertex */
            cx,        /* IN: Current cell x */
            curr_cy,   /* IN: Current cell y */
            d_grid,    /* IN: grid object */
            scaling,   /* IN: Scaling */
            d_base_map /* IN: Map to test */
        );
        if (beste != nullptr) {
          closest_eid = beste->eid;
          break;
        }
      }
      d_closest_eids[point_idx] = closest_eid;
    });
  }

  std::shared_ptr<grid_t> get_grid() { return grid_; }

 private:
  std::shared_ptr<grid_t> grid_;
};
}  // namespace rayjoin
#endif  // APP_PIP_GRID_H
