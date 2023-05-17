#ifndef RAYJOIN_RT_PRIMITIVE_H
#define RAYJOIN_RT_PRIMITIVE_H
#include <thrust/device_vector.h>

#include "config.h"
#include "util/array_view.h"
#include "util/helpers.h"
#include "util/stream.h"
#include "util/util.h"

namespace rayjoin {
template <typename MAP_T, typename SCALING_T>
void FillPrimitivesTriangle(Stream& stream, const MAP_T& d_map,
                            const SCALING_T& scaling,
                            thrust::device_vector<float3>& triangle_points) {
  auto ne = d_map.get_edges_num();

  triangle_points.resize(ne * 3);

  ArrayView<float3> d_triangle_points(triangle_points);

  ForEach(stream, ne, [=] __device__(uint32_t eid) mutable {
    const auto& e = d_map.get_edge(eid);
    auto p1_idx = e.p1_idx;
    auto p2_idx = e.p2_idx;
    auto p1 = d_map.get_point(p1_idx);
    auto p2 = d_map.get_point(p2_idx);
    float x1, x2, y1, y2;

    if (e.b == 0) {
      assert(p1.x == p2.x);
      if (p1.y > p2.y) {
        SWAP(p1, p2);
      }
      x1 = x2 = scaling.UnscaleX(p1.x);

      y1 = next_float_from_double(scaling.UnscaleY(p1.y), -1, ROUNDING_ITER);
      y2 = next_float_from_double(scaling.UnscaleY(p2.y), 1, ROUNDING_ITER);
    } else {
      assert(p1.x != p2.x);
      if (p1.x > p2.x) {
        SWAP(p1, p2);
      }

      // use double is much faster than rational
      // this does not need to be accurate
      // use double is much faster than rational
      // this does not need to be accurate
      double a = (double) -e.a / e.b;
      double b = (double) -e.c / e.b;

      x1 = next_float_from_double(scaling.UnscaleX(p1.x), -1, ROUNDING_ITER);
      y1 = scaling.UnscaleY(a * scaling.ScaleX(x1) + b);

      x2 = next_float_from_double(scaling.UnscaleX(p2.x), 1, ROUNDING_ITER);
      y2 = scaling.UnscaleY(a * scaling.ScaleX(x2) + b);
    }

    d_triangle_points[eid * 3] = {x1, y1, PRIMITIVE_HEIGHT};
    d_triangle_points[eid * 3 + 1] = {x2, y2, PRIMITIVE_HEIGHT};
    d_triangle_points[eid * 3 + 2] = {(x1 + x2) / 2, (y1 + y2) / 2,
                                      PRIMITIVE_HEIGHT};

    if (eid == 0) {
      printf("double eid: %u %.6lf %.6lf %.6lf %.6lf\n", eid,
             scaling.UnscaleX(p1.x), scaling.UnscaleY(p1.y),
             scaling.UnscaleX(p2.x), scaling.UnscaleY(p2.y));

      printf("float eid: %u %.6f %.6f %.6f %.6f\n", eid, x1, y1, x2, y2);
    }
  });
}

template <typename MAP_T, typename SCALING_T>
void FillPrimitives(Stream& stream, const MAP_T& d_map,
                    const SCALING_T& scaling,
                    thrust::device_vector<OptixAabb>& aabbs) {
  auto ne = d_map.get_edges_num();
  aabbs.resize(ne);
  ArrayView<OptixAabb> d_aabbs(aabbs);

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

    aabb.minX = next_float_from_double(min(x1, x2), -1, ROUNDING_ITER);
    aabb.maxX = next_float_from_double(max(x1, x2), 1, ROUNDING_ITER);
    aabb.minY = next_float_from_double(min(y1, y2), -1, ROUNDING_ITER);
    aabb.maxY = next_float_from_double(max(y1, y2), 1, ROUNDING_ITER);
    aabb.minZ = -PRIMITIVE_HEIGHT / 2;
    aabb.maxZ = PRIMITIVE_HEIGHT / 2;
  });
}

template <typename MAP_T, typename SCALING_T>
void FillPrimitivesSketch(Stream& stream, const MAP_T& d_map,
                          const SCALING_T& scaling,
                          thrust::device_vector<OptixAabb>& aabbs) {
  int sketch_size = SKETCH_SIZE;
  auto ne = d_map.get_edges_num();
  auto n_sketch = (ne + sketch_size - 1) / sketch_size;

  aabbs.resize(n_sketch);

  ArrayView<OptixAabb> d_aabbs(aabbs);

  ForEach(stream, n_sketch, [=] __device__(size_t sketch_id) mutable {
    size_t eid_begin = std::min(sketch_id * sketch_size, ne);
    size_t eid_end = std::min((sketch_id + 1) * sketch_size, ne);
    auto& aabb = d_aabbs[sketch_id];

    aabb.minX = std::numeric_limits<float>::max();
    aabb.maxX = std::numeric_limits<float>::lowest();
    aabb.minY = std::numeric_limits<float>::max();
    aabb.maxY = std::numeric_limits<float>::lowest();
    aabb.minZ = -PRIMITIVE_HEIGHT / 2;
    aabb.maxZ = PRIMITIVE_HEIGHT / 2;

    for (size_t eid = eid_begin; eid < eid_end; eid++) {
      const auto& e = d_map.get_edge(eid);
      auto p1_idx = e.p1_idx;
      auto p2_idx = e.p2_idx;
      const auto& p1 = d_map.get_point(p1_idx);
      const auto& p2 = d_map.get_point(p2_idx);
      auto x1 = scaling.UnscaleX(p1.x);
      auto y1 = scaling.UnscaleY(p1.y);
      auto x2 = scaling.UnscaleX(p2.x);
      auto y2 = scaling.UnscaleY(p2.y);

      aabb.minX = std::min(aabb.minX, (float) std::min(x1, x2));
      aabb.maxX = std::max(aabb.maxX, (float) std::max(x1, x2));
      aabb.minY = std::min(aabb.minY, (float) std::min(y1, y2));
      aabb.maxY = std::max(aabb.maxY, (float) std::max(y1, y2));
    }
    aabb.minX = next_float_from_double(aabb.minX, -1, ROUNDING_ITER);
    aabb.maxX = next_float_from_double(aabb.maxX, 1, ROUNDING_ITER);
    aabb.minY = next_float_from_double(aabb.minY, -1, ROUNDING_ITER);
    aabb.maxY = next_float_from_double(aabb.maxY, 1, ROUNDING_ITER);
  });
}

}  // namespace rayjoin

#endif  // RAYJOIN_RT_PRIMITIVE_H
