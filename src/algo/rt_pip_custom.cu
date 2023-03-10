#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "algo/launch_parameters.h"
#include "config.h"
#include "util/helpers.h"
#include "util/int128_intrinsics.h"
#include "util/rational.h"
#include "util/util.h"
#include "util/vec_math.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

extern "C" __constant__ rayjoin::LaunchParamsPIP params;

extern "C" __global__ void __anyhit__pip_custom() { optixTerminateRay(); }

extern "C" __global__ void __intersection__pip_custom() {
  float3 ray_orig = optixGetWorldRayOrigin();
  auto point_idx = optixGetPayload_0();
  using internal_coord_t = typename rayjoin::LaunchParamsPIP::internal_coord_t;
  auto init_best_y = std::numeric_limits<internal_coord_t>::max();
  internal_coord_t best_y;
  uint2 best_y_storage{optixGetPayload_1(), optixGetPayload_2()};
  double best_e_slope;
  uint2 best_e_slope_storage{optixGetPayload_3(), optixGetPayload_4()};
  auto eid = optixGetPrimitiveIndex();
  auto query_map_id = params.query_map_id;
  const auto& scaling = params.scaling;
  const auto& e = params.dst_edges[eid];
  const auto& p1 = params.dst_points[e.p1_idx];
  const auto& p2 = params.dst_points[e.p2_idx];
  auto x_min = min(p1.x, p2.x);
  auto x_max = max(p1.x, p2.x);
  const auto& src_p = params.src_points[point_idx];
  auto x_src_p = src_p.x;
  auto y_src_p = src_p.y;

  unpack64(best_y_storage.x, best_y_storage.y, &best_y);
  unpack64(best_e_slope_storage.x, best_e_slope_storage.y, &best_e_slope);
#ifndef NDEBUG
  params.hit_count[point_idx]++;
#endif
  /*
   * Is point outside x bounds of this edge?  Use simulation of simplicity:
   * shift map 1 by epsilon relative to map 0.  This also eliminates vertical
   * edges.
   */
  if (x_src_p < x_min || x_src_p > x_max ||
      x_src_p == ((query_map_id == 0) ? x_min : x_max)) {
    return;
  }

  assert(e.b != 0);

  auto xsect_y = (double) (-e.a * x_src_p - e.c) / e.b;
  auto diff_y = xsect_y - y_src_p;

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
  if (diff_y < 0) {
#ifndef NDEBUG
    params.above_edge_count[point_idx]++;
#endif
    return;
  }

  if (xsect_y > best_y) {
    auto deviant = scaling.UnscaleY(xsect_y) - scaling.UnscaleY(best_y);
    // continue search only if current primitive is not deviant too much
    if (best_y != init_best_y && deviant > params.early_term_deviant) {
      optixReportIntersection(0, 0);  // terminate ray
    }
#ifndef NDEBUG
    params.fail_update_count[point_idx]++;
#endif
    return;
  }
  double current_e_slope = (double) e.a / e.b;

  if (xsect_y == best_y) {
    bool flag = current_e_slope > best_e_slope;

    /* If im==0 we want the bigger slope, if im==1, the smaller. */

    if ((query_map_id && !flag) || (flag && !query_map_id)) {
      return;
    }
  }

  best_y = xsect_y;

  pack64(&best_y, best_y_storage.x, best_y_storage.y);
  optixSetPayload_1(best_y_storage.x);
  optixSetPayload_2(best_y_storage.y);
  pack64(&current_e_slope, best_e_slope_storage.x, best_e_slope_storage.y);
  optixSetPayload_3(best_e_slope_storage.x);
  optixSetPayload_4(best_e_slope_storage.y);

  rayjoin::polygon_id_t ipol = 0;

  if (p1.x < p2.x) {
    ipol = e.right_polygon_id;
  } else {
    ipol = e.left_polygon_id;
  }
  params.point_in_polygon[point_idx] = ipol;

#ifndef NDEBUG
  params.closer_count[point_idx]++;
#endif
}

extern "C" __global__ void __raygen__pip_custom() {
  float3 ray_dir = {0, 1, 0};
  const auto& src_points = params.src_points;
  const auto& scaling = params.scaling;

  for (unsigned int point_idx = OPTIX_TID_1D; point_idx < src_points.size();
       point_idx += OPTIX_TOTAL_THREADS_1D) {
    // FIXME: Shoot two rays
    auto& p = src_points[point_idx];
    float3 ray_origin = {(float) scaling.UnscaleX(p.x),
                         (float) scaling.UnscaleY(p.y), 0};
    float tmin = 0;
    float tmax = RAY_TMAX;
    auto best_y = std::numeric_limits<
        typename rayjoin::LaunchParamsPIP::internal_coord_t>::max();
    static_assert(sizeof(best_y) == 8,
                  "Invalid internal coordinate type");  // current we use int64
    uint2 best_y_storage;
    pack64(&best_y, best_y_storage.x, best_y_storage.y);

    double best_e_slope;
    uint2 best_e_slope_storage;
    pack64(&best_e_slope, best_e_slope_storage.x, best_e_slope_storage.y);

    params.point_in_polygon[point_idx] = EXTERIOR_FACE_ID;

    optixTrace(params.traversable, ray_origin, ray_dir,
               tmin,  // tmin
               tmax,  // tmax
               0,     // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               point_idx, best_y_storage.x, best_y_storage.y,
               best_e_slope_storage.x, best_e_slope_storage.y);
  }
}
