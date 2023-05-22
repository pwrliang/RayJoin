
#ifndef RAYJOIN_TREE_PRIMTIVE_H
#define RAYJOIN_TREE_PRIMTIVE_H
#include "lbvh.cuh"
#include "util/array_view.h"
#include "util/helpers.h"
#include "util/stream.h"

namespace rayjoin {
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

    min_x = next_float_from_double(min_x, -1, ROUNDING_ITER);
    max_x = next_float_from_double(max_x, 1, ROUNDING_ITER);
    min_y = next_float_from_double(min_y, -1, ROUNDING_ITER);
    max_x = next_float_from_double(max_x, 1, ROUNDING_ITER);

    box.lower = make_float4(min_x, min_y, 0, 0);
    box.upper = make_float4(max_x, max_y, 0, 0);
    return box;
  }
};

template <typename MAP_T, typename SCALING_T, typename SEGMENT_T>
void FillPrimitivesLBVH(Stream& stream, const MAP_T& d_map,
                        const SCALING_T& scaling,
                        pinned_vector<SEGMENT_T>& primitives) {
  primitives.resize(d_map.get_edges_num());

  ForEach(
      stream, d_map.get_edges_num(),
      [=] __device__(size_t eid, ArrayView<SEGMENT_T> edges) {
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

        edges[eid].p1 = {(float) x1, (float) y1};
        edges[eid].p2 = {(float) x2, (float) y2};
      },
      ArrayView<SEGMENT_T>(primitives));
}

}  // namespace rayjoin

#endif  // RAYJOIN_TREE_PRIMTIVE_H
