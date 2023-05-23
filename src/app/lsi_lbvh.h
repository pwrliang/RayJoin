#ifndef APP_LSI_LBVH_H
#define APP_LSI_LBVH_H

#include <memory>

#include "algo/lsi.h"
#include "app/lsi.h"
#include "app/query_config.h"
#include "lbvh.cuh"
#include "tree/primtive.h"
#include "util/helpers.h"

namespace rayjoin {
template <typename CONTEXT_T>
class LSILBVH : public LSI<CONTEXT_T> {
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using lsi = LSI<CONTEXT_T>;
  using coord_t = typename lsi::coord_t;
  using internal_coord_t = typename lsi::internal_coord_t;
  using xsect_t = typename lsi::xsect_t;

 public:
  explicit LSILBVH(CONTEXT_T& ctx) : LSI<CONTEXT_T>(ctx) {}

  void Init(size_t max_n_xsects) override { lsi::Init(max_n_xsects); }

  void Query(Stream& stream, int query_map_id) override {
    auto& ctx = this->ctx_;
    int base_map_id = 1 - query_map_id;
    auto d_query_map = ctx.get_map(query_map_id)->DeviceObject();
    auto d_base_map = ctx.get_map(base_map_id)->DeviceObject();
    auto scaling = ctx.get_scaling();
    const auto bvh_dev = config_.lbvh->get_device_repr();
    auto d_xsect_queue = this->xsect_queue_.DeviceObject();

    this->xsect_queue_.Clear(stream);
#ifndef NDEBUG
    this->prof_counter_.set(0, stream);
    visited_nodes_.set(0, stream);
    auto* d_prof_counter = this->prof_counter_.data();
    auto* d_visited_nodes = visited_nodes_.data();
#endif

    ForEach(stream, d_query_map.get_edges_num(),
            [=] __device__(size_t eid1) mutable {
              const auto& e1 = d_query_map.get_edge(eid1);
              const auto& e1_p1 = d_query_map.get_point(e1.p1_idx);
              const auto& e1_p2 = d_query_map.get_point(e1.p2_idx);
              auto x1 = scaling.UnscaleX(e1_p1.x);
              auto y1 = scaling.UnscaleY(e1_p1.y);
              auto x2 = scaling.UnscaleX(e1_p2.x);
              auto y2 = scaling.UnscaleY(e1_p2.y);

              auto min_x = min(x1, x2);
              auto max_x = max(x1, x2);
              auto min_y = min(y1, y2);
              auto max_y = max(y1, y2);

              lbvh::aabb<float> box;
              box.upper = make_float4(max_x, max_y, 0.0, 0.0);
              box.lower = make_float4(min_x, min_y, 0.0, 0.0);

              auto pair = lbvh::query_device_all(
                  bvh_dev, lbvh::overlaps(box),
                  [=] __device__(std::uint32_t eid2) mutable {
                    const auto& e2 = d_base_map.get_edge(eid2);
                    const auto& e2_p1 = d_base_map.get_point(e2.p1_idx);
                    const auto& e2_p2 = d_base_map.get_point(e2.p2_idx);
                    tcb::rational<coefficient_t> xsect_x, xsect_y;

                    if (dev::intersect_test(e1, e1_p1, e1_p2, e2, e2_p1, e2_p2,
                                            xsect_x, xsect_y)) {
                      dev::Intersection<internal_coord_t> xsect;
                      xsect.x = xsect_x;
                      xsect.y = xsect_y;
                      xsect.eid[0] = eid1;
                      xsect.eid[1] = eid2;
                      d_xsect_queue.AppendWarp(xsect);
                    }
                  });
              auto traversed_aabbs = pair.first;
              auto num_found = pair.second;
#ifndef NDEBUG
              atomicAdd((unsigned long long int*) d_prof_counter, num_found);
              atomicAdd((unsigned long long int*) d_visited_nodes,
                        traversed_aabbs);
#endif
            });
    stream.Sync();

#ifndef NDEBUG
    LOG(INFO) << "Total tests: " << this->prof_counter_.get(stream);
    LOG(INFO) << "Visited nodes: " << visited_nodes_.get(stream);
#endif
    stream.Sync();
  }

  void set_config(const QueryConfigLBVH& config) { config_ = config; }

 private:
  SharedValue<uint64_t> visited_nodes_;
  QueryConfigLBVH config_;
};
}  // namespace rayjoin

#endif  // APP_LSI_LBVH_H
