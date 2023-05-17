#ifndef APP_LSI_RT_H
#define APP_LSI_RT_H

#include <thrust/sort.h>
#include <thrust/unique.h>

#include <utility>

#include "algo/lsi.h"
#include "app/lsi.h"
#include "app/rt_query_config.h"
#include "rt/rt_engine.h"
#include "util/helpers.h"

namespace rayjoin {
template <typename CONTEXT_T>
class LSIRT : public LSI<CONTEXT_T> {
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using lsi = LSI<CONTEXT_T>;
  using coord_t = typename lsi::coord_t;
  using internal_coord_t = typename lsi::internal_coord_t;
  using xsect_t = typename lsi::xsect_t;

 public:
  explicit LSIRT(CONTEXT_T& ctx, const std::shared_ptr<RTEngine>& rt_engine)
      : LSI<CONTEXT_T>(ctx), rt_engine_(rt_engine) {}

  void set_query_config(RTQueryConfig config) { config_ = config; }

  void Init(size_t max_n_xsects) override { lsi::Init(max_n_xsects); }

  void BuildIndex(int query_map_id) override {
    auto& ctx = this->ctx_;
    auto& stream = ctx.get_stream();
    const auto& scaling = ctx.get_scaling();
    int base_map_id = 1 - query_map_id;
    auto d_map = ctx.get_map(base_map_id)->DeviceObject();
    auto ne = d_map.get_edges_num();
    auto epsilon = 0.0001;  // FIXME
    auto rounding_iter = config_.rounding_iter;

    if (config_.use_triangle) {
      auto bb = ctx.get_bounding_box();
      auto mid_x = (bb.min_x + bb.max_x) / 2.0;
      auto mid_y = (bb.min_y + bb.max_y) / 2.0;
      auto delta_x = scaling.ScaleX(mid_x + epsilon) - scaling.ScaleX(mid_x);
      auto delta_y = scaling.ScaleY(mid_y + epsilon) - scaling.ScaleY(mid_y);

      triangle_points_.resize(d_map.get_edges_num() * 3);

      ArrayView<float3> d_triangle_points(triangle_points_);

      ForEach(stream, ne, [=] __device__(uint32_t eid) mutable {
        const auto& e = d_map.get_edge(eid);
        auto p1_idx = e.p1_idx;
        auto p2_idx = e.p2_idx;
        auto p1 = d_map.get_point(p1_idx);
        auto p2 = d_map.get_point(p2_idx);
        auto x1 = scaling.UnscaleX(p1.x);
        auto y1 = scaling.UnscaleY(p1.y);
        auto x2 = scaling.UnscaleX(p2.x);
        auto y2 = scaling.UnscaleY(p2.y);

        if (e.b == 0) {
          assert(p1.x == p2.x);
          if (p1.y > p2.y) {
            SWAP(p1, p2);
          }

          x1 = x2 = scaling.UnscaleX(p1.x);
          y1 = scaling.UnscaleY(p1.y - delta_y);
          y2 = scaling.UnscaleY(p2.y + delta_y);
        } else {
          assert(p1.x != p2.x);
          if (p1.x > p2.x) {
            SWAP(p1, p2);
          }

          // use double is much faster than rational
          // this does not need to be accurate
          double a = -e.a / e.b;
          double b = -e.c / e.b;

          auto new_x1 = p1.x - delta_x;
          auto new_y1 = a * new_x1 + b;

          auto new_x2 = p2.x + delta_x;
          auto new_y2 = a * new_x2 + b;

          x1 = scaling.UnscaleX(new_x1);
          y1 = scaling.UnscaleY(new_y1);
          x2 = scaling.UnscaleX(new_x2);
          y2 = scaling.UnscaleY(new_y2);
        }

        d_triangle_points[eid * 3] = {(float) x1, (float) y1, 0};
        d_triangle_points[eid * 3 + 1] = {(float) x2, (float) y2, 0};
        d_triangle_points[eid * 3 + 2] = {
            (float) (x1 + x2) / 2, (float) (y1 + y2) / 2, PRIMITIVE_HEIGHT};
      });
      handles_[query_map_id] =
          rt_engine_->BuildAccelTriangles(stream, d_triangle_points);
    } else {
      aabbs_.resize(ne);

      ArrayView<OptixAabb> d_aabbs(aabbs_);

      ForEach(stream, ne, [=] __device__(size_t eid) mutable {
        const auto& e = d_map.get_edge(eid);
        auto p1_idx = e.p1_idx;
        auto p2_idx = e.p2_idx;
        const auto& p1 = d_map.get_point(p1_idx);
        const auto& p2 = d_map.get_point(p2_idx);
        auto x1 = scaling.UnscaleX(p1.x);
        auto y1 = scaling.UnscaleY(p1.y);
        auto x2 = scaling.UnscaleX(p2.x);
        auto y2 = scaling.UnscaleY(p2.y);
        auto& aabb = d_aabbs[eid];

        aabb.minX = next_float_from_double(min(x1, x2), -1, rounding_iter);
        aabb.maxX = next_float_from_double(max(x1, x2), 1, rounding_iter);
        aabb.minY = next_float_from_double(min(y1, y2), -1, rounding_iter);
        aabb.maxY = next_float_from_double(max(y1, y2), 1, rounding_iter);
        aabb.minZ = -PRIMITIVE_HEIGHT / 2;
        aabb.maxZ = PRIMITIVE_HEIGHT / 2;
      });
      handles_[query_map_id] = rt_engine_->BuildAccelCustom(stream, d_aabbs);
    }

    stream.Sync();
    if (config_.fau) {
      if (config_.use_triangle) {
        triangle_points_.resize(0);
        triangle_points_.shrink_to_fit();
      } else {
        aabbs_.resize(0);
        aabbs_.shrink_to_fit();
      }
    }
  }

  ArrayView<xsect_t> Query(int query_map_id) override {
    auto& ctx = this->ctx_;
    int base_map_id = 1 - query_map_id;
    auto d_query_map = ctx.get_map(query_map_id)->DeviceObject(),
         d_base_map = ctx.get_map(base_map_id)->DeviceObject();
    auto scaling = ctx.get_scaling();
    auto& stream = ctx.get_stream();
    auto& xsects = this->xsect_edges_;

    LaunchParamsLSI params;
    auto module_id = config_.use_triangle
                         ? ModuleIdentifier::MODULE_ID_LSI
                         : ModuleIdentifier::MODULE_ID_LSI_CUSTOM;
    params.query_map_id = query_map_id;
    params.scaling = scaling;
    params.base_edges = d_base_map.get_edges().data();
    params.base_points = d_base_map.get_points().data();
    params.query_edges = d_query_map.get_edges();
    params.query_points = d_query_map.get_points().data();
    params.traversable = handles_[query_map_id];
    params.rounding_iter = config_.rounding_iter;
    params.xsects = xsects.DeviceObject();

    xsects.Clear(stream);

    rt_engine_->CopyLaunchParams(stream, params);

    rt_engine_->Render(
        stream, module_id,
        dim3{(unsigned int) ctx.get_map(query_map_id)->get_edges_num(), 1, 1});

    size_t n_xsects = xsects.size(stream);

    ForEach(
        stream, n_xsects,
        [=] __device__(size_t idx, ArrayView<xsect_t> xsects) mutable {
          auto& xsect = xsects[idx];
          auto base_eid = xsect.eid[0], query_eid = xsect.eid[1];

          const auto& base_e = d_base_map.get_edge(base_eid);
          const auto& base_e_p1 = d_base_map.get_point(base_e.p1_idx);
          const auto& base_e_p2 = d_base_map.get_point(base_e.p2_idx);

          const auto& query_e = d_query_map.get_edge(query_eid);
          const auto& query_e_p1 = d_query_map.get_point(query_e.p1_idx);
          const auto& query_e_p2 = d_query_map.get_point(query_e.p2_idx);

          auto denom = (coefficient_t) base_e.a * query_e.b -
                       (coefficient_t) query_e.a * base_e.b;
          auto numx = (coefficient_t) query_e.c * base_e.b -
                      (coefficient_t) base_e.c * query_e.b;
          auto numy = (coefficient_t) query_e.a * base_e.c -
                      (coefficient_t) base_e.a * query_e.c;

          tcb::rational<coefficient_t> xsect_x(numx, denom);
          tcb::rational<coefficient_t> xsect_y(numy, denom);

          auto t = MIN4(base_e_p1.x, base_e_p2.x, query_e_p1.x, query_e_p2.x);
          if (xsect_x < t) {
            xsect_x = t;
          }

          t = MAX4(base_e_p1.x, base_e_p2.x, query_e_p1.x, query_e_p2.x);
          if (xsect_x > t) {
            xsect_x = t;
          }

          t = MIN4(base_e_p1.y, base_e_p2.y, query_e_p1.y, query_e_p2.y);
          if (xsect_y < t) {
            xsect_y = t;
          }
          t = MAX4(base_e_p1.y, base_e_p2.y, query_e_p1.y, query_e_p2.y);
          if (xsect_y > t) {
            xsect_y = t;
          }
          xsect.x = xsect_x;
          xsect.y = xsect_y;
        },
        ArrayView<xsect_t>(xsects.data(), n_xsects));

    stream.Sync();

#ifndef NDEBUG
    LOG(INFO) << "Total Tests: " << n_xsects;
#endif
    return ArrayView<xsect_t>(this->xsect_edges_.data(),
                              this->xsect_edges_.size(stream));
  }

 private:
  std::shared_ptr<RTEngine> rt_engine_;
  RTQueryConfig config_;
  std::map<int, OptixTraversableHandle> handles_;
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<float3> triangle_points_;
};
}  // namespace rayjoin

#endif  // APP_LSI_RT_H
