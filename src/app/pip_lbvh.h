
#ifndef APP_PIP_LBVH_H
#define APP_PIP_LBVH_H
#include <memory>
#include <utility>

#include "algo/pip.h"
#include "app/pip.h"
#include "config.h"
#include "lbvh.cuh"
#include "tree/primtive.h"

namespace rayjoin {
template <typename CONTEXT_T>
class PIPLBVH : public PIP<CONTEXT_T> {
  using map_t = typename CONTEXT_T::map_t;
  using point_t = typename map_t::point_t;
  using coefficient_t = typename map_t::coefficient_t;

 public:
  explicit PIPLBVH(CONTEXT_T& ctx) : PIP<CONTEXT_T>(ctx) {}

  void set_config(QueryConfigLBVH config) { config_ = std::move(config); }

  void Query(Stream& stream, int query_map_id,
             ArrayView<point_t> d_query_points) {
    auto base_map_id = 1 - query_map_id;
    auto& scaling = this->ctx_.get_scaling();
    auto d_base_map = this->ctx_.get_map(base_map_id)->DeviceObject();
    auto n_points = d_query_points.size();
    const auto bvh_dev = config_.lbvh->get_device_repr();

    this->closest_eids_.resize(n_points);

    ArrayView<index_t> d_closest_eids(this->closest_eids_);

    ForEach(stream, n_points, [=] __device__(size_t point_idx) mutable {
      const auto& p = d_query_points[point_idx];
      // init value means point is not in hit
      auto closest_eid = static_cast<index_t>(DONTKNOW);
      auto x = scaling.UnscaleX(p.x);
      auto y = scaling.UnscaleY(p.y);
      double best_y = std::numeric_limits<double>::infinity();
      const dev::Edge<coefficient_t>* best_e = nullptr;

      lbvh::aabb<float> box;
      box.upper = make_float4(x, std::numeric_limits<float>::max(), 0.0, 0.0);
      box.lower = make_float4(x, next_float_from_double(y, -1, ROUNDING_ITER),
                              0.0, 0.0);

      auto pair = lbvh::query_device_all(
          bvh_dev, lbvh::overlaps(box),
          [&] __device__(std::uint32_t eid) mutable {
            auto& e = d_base_map.get_edge(eid);
            auto& p1 = d_base_map.get_point(e.p1_idx);
            auto& p2 = d_base_map.get_point(e.p2_idx);
            auto x_min = min(p1.x, p2.x);
            auto x_max = max(p1.x, p2.x);
            auto x_src_p = p.x;
            auto y_src_p = p.y;
            /*
             * Is point outside x bounds of this edge?  Use simulation of
             * simplicity: shift map 1 by epsilon relative to map 0.  This also
             * eliminates vertical edges.
             */
            if (x_src_p < x_min || x_src_p > x_max ||
                x_src_p == ((query_map_id == 0) ? x_min : x_max)) {
              return;
            }

            assert(e.b != 0);
            /*
             * Is point above edge?  If point is on edge, use simulation of
             * simplicity.
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
              return;
            }

            if (xsect_y > best_y) {
              return;
            }

            if (xsect_y ==
                best_y) { /* Line up from point cuts this edge and another
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
                return;
            }
            best_y = xsect_y;
            best_e = &e;
          });
      auto traversed_aabbs = pair.first;
      auto num_found = pair.second;

      if (best_e != nullptr) {
        closest_eid = best_e->eid;
      }
      d_closest_eids[point_idx] = closest_eid;
    });
  }

 private:
  QueryConfigLBVH config_;
};
}  // namespace rayjoin
#endif  // APP_PIP_LBVH_H
