#ifndef APP_LSI_LBVH_H
#define APP_LSI_LBVH_H

#include "algo/lsi.h"
#include "app/lsi.h"
#include "lbvh.cuh"

namespace rayjoin {
template <typename CONTEXT_T>
class LSILBVH : public LSI<CONTEXT_T> {
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using lsi = LSI<CONTEXT_T>;
  using coord_t = typename lsi::coord_t;
  using internal_coord_t = typename lsi::internal_coord_t;
  using xsect_t = typename lsi::xsect_t;

 public:
  struct segment {
    float2 p1, p2;
  };

  struct aabb_getter {
    __device__ lbvh::aabb<float> operator()(const segment& f) const noexcept {
      lbvh::aabb<float> box;
      auto min_x = min(f.p1.x, f.p2.x);
      auto max_x = max(f.p1.x, f.p2.x);
      auto min_y = min(f.p1.y, f.p2.y);
      auto max_y = max(f.p1.y, f.p2.y);

      box.lower = make_float4(min_x, min_y, 0, 0);
      box.upper = make_float4(max_x, max_y, 0, 0);
      return box;
    }
  };

  explicit LSILBVH(CONTEXT_T& ctx, bool profiling)
      : LSI<CONTEXT_T>(ctx), profiling_(profiling) {}

  void Init(size_t max_n_xsects) override {
    lsi::Init(max_n_xsects);

    auto& ctx = this->ctx_;
    auto scaling = ctx.get_scaling();
    auto bb = ctx.get_bounding_box();
    auto mid_x = (bb.min_x + bb.max_x) / 2.0;
    auto& stream = ctx.get_stream();
    pinned_vector<segment> primitives;
    int base_map_id = 0;

    auto map = ctx.get_map(base_map_id);
    auto d_map = map->DeviceObject();

    primitives.resize(map->get_edges_num());

    CHECK_GT(primitives.size(), 1)
        << "At least two primitives are required to use LBVH";

    ForEach(
        stream, map->get_edges_num(),
        [=] __device__(size_t eid, ArrayView<segment> edges) {
          const auto& e = d_map.get_edge(eid);
          auto p1 = d_map.get_point(e.p1_idx);
          auto p2 = d_map.get_point(e.p2_idx);
          if (p1.x > p2.x) {
            SWAP(p1, p2);
          }
          auto x1 = scaling.UnscaleX(p1.x);
          auto y1 = scaling.UnscaleY(p1.y);
          auto x2 = scaling.UnscaleX(p2.x);
          auto y2 = scaling.UnscaleY(p2.y);

          //            if (e.b != 0) {
          //              assert(p1.x < p2.x);
          //              // use double is much faster than rational
          //              // this does not need to be accurate
          //              double a = -e.a / e.b;
          //              double b = -e.c / e.b;
          //
          //              auto new_x1 = p1.x - margin;
          //              auto new_y1 = a * new_x1 + b;
          //
          //              auto new_x2 = p2.x + margin;
          //              auto new_y2 = a * new_x2 + b;
          //
          //              x1 = scaling.UnscaleX(new_x1);
          //              y1 = scaling.UnscaleY(new_y1);
          //              x2 = scaling.UnscaleX(new_x2);
          //              y2 = scaling.UnscaleY(new_y2);
          //            }

          edges[eid].p1 = {(float) x1, (float) y1};
          edges[eid].p2 = {(float) x2, (float) y2};
        },
        ArrayView<segment>(primitives));
    stream.Sync();
    // copy primitives to the GPU
    bvh_.assign(primitives);
  }

  void BuildIndex(int query_map_id) override { bvh_.construct(profiling_); }

  ArrayView<xsect_t> Query(int query_map_id) override {
    auto& ctx = this->ctx_;
    int base_map_id = 1 - query_map_id;
    auto d_query_map = ctx.get_map(query_map_id)->DeviceObject();
    auto d_base_map = ctx.get_map(base_map_id)->DeviceObject();
    auto scaling = ctx.get_scaling();
    auto& stream = ctx.get_stream();
    const auto bvh_dev = bvh_.get_device_repr();
    auto d_xsects = this->xsect_edges_.DeviceObject();

    this->xsect_edges_.Clear(stream);
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
                      d_xsects.AppendWarp(xsect);
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
    return ArrayView<xsect_t>(this->xsect_edges_.data(),
                              this->xsect_edges_.size(stream));
  }

 private:
  bool profiling_;
  SharedValue<uint64_t> visited_nodes_;
  lbvh::bvh<float, segment, aabb_getter> bvh_;
};
}  // namespace rayjoin

#endif  // APP_LSI_LBVH_H
