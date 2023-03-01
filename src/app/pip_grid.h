
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

  thrust::device_vector<polygon_id_t>& Query(int map_id,
                                             ArrayView<point_t> d_points) {
    auto& stream = this->ctx_.get_stream();
    auto& scaling = this->ctx_.get_scaling();
    auto d_grid = grid_->DeviceObject();
    auto d_map = this->ctx_.get_map(map_id)->DeviceObject();
    auto gsize = grid_->get_grid_size();

    this->polygon_ids_.resize(d_points.size());
    thrust::fill(thrust::cuda::par.on(stream.cuda_stream()),
                 this->polygon_ids_.begin(), this->polygon_ids_.end(),
                 DONTKNOW);

    ArrayView<polygon_id_t> d_point_in_polygon(this->polygon_ids_);

    ForEach(stream, d_points.size(), [=] __device__(size_t point_idx) mutable {
      polygon_id_t ipol = 0;  // if not hit any edges, vertex is on face0
      const auto& p = d_points[point_idx];
      auto cx = dev::calculate_cell(gsize, scaling, p.x);
      auto cy = dev::calculate_cell(gsize, scaling, p.y);

      assert(cx >= 0 && cx < gsize);
      assert(cy >= 0 && cy < gsize);

      // shoot a ray toward positive y-axis
      for (auto curr_cy = cy; curr_cy < gsize; curr_cy++) {
        // find a closest edge of dst-map to this vertex in
        // this cell
        auto* beste = dev::test_against_edges_of_this_cell(
            p,       /* IN: current vertex */
            cx,      /* IN: Current cell x */
            curr_cy, /* IN: Current cell y */
            d_grid,  /* IN: grid object */
            scaling, /* IN: Scaling */
            d_map    /* IN: Map to test */
        );
        if (beste != nullptr) {
          if (d_map.get_point(beste->p1_idx).x <
              d_map.get_point(beste->p2_idx).x) {
            ipol = beste->right_polygon_id;
          } else {
            ipol = beste->left_polygon_id;
          }
          break;
        }
      }
      d_point_in_polygon[point_idx] = ipol;
    });
    stream.Sync();
    return this->polygon_ids_;
  }

  std::shared_ptr<grid_t> get_grid() { return grid_; }

 private:
  std::shared_ptr<grid_t> grid_;
};
}  // namespace rayjoin
#endif  // APP_PIP_GRID_H
