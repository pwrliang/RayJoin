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
    auto win_size = config_.win_size;
    auto area_enlarge = config_.enlarge;
    FillPrimitivesGroup(stream, d_map, scaling, win_size, area_enlarge, aabbs_,
                        eid_range_);

    handles_[query_map_id] =
        rt_engine_->BuildAccelCustom(stream, ArrayView<OptixAabb>(aabbs_));

    stream.Sync();
    if (config_.fau) {
      aabbs_.resize(0);
      aabbs_.shrink_to_fit();
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
    auto module_id = ModuleIdentifier::MODULE_ID_LSI_CUSTOM;
    params.query_map_id = query_map_id;
    params.scaling = scaling;
    params.base_edges = d_base_map.get_edges().data();
    params.base_points = d_base_map.get_points().data();
    params.eid_range = thrust::raw_pointer_cast(eid_range_.data());
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
  thrust::device_vector<thrust::pair<size_t, size_t>> eid_range_;
};
}  // namespace rayjoin

#endif  // APP_LSI_RT_H
