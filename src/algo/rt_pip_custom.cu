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
  using coefficient_t = rayjoin::coefficient_t;
  float3 ray_orig = optixGetWorldRayOrigin();
  auto point_idx = optixGetPayload_0();
  using internal_coord_t = typename rayjoin::LaunchParamsPIP::internal_coord_t;
  auto init_best_y = std::numeric_limits<internal_coord_t>::max();
  double best_y;
  uint2 best_y_storage{optixGetPayload_1(), optixGetPayload_2()};
  auto eid = optixGetPrimitiveIndex();
  auto query_map_id = params.query_map_id;
  const auto& scaling = params.scaling;
  const auto& e = params.base_map_edges[eid];
  const auto& p1 = params.base_map_points[e.p1_idx];
  const auto& p2 = params.base_map_points[e.p2_idx];
  auto x_min = min(p1.x, p2.x);
  auto x_max = max(p1.x, p2.x);
  const auto& src_p = params.query_points[point_idx];
  auto x_src_p = src_p.x;
  auto y_src_p = src_p.y;

  unpack64(best_y_storage.x, best_y_storage.y, &best_y);
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
#ifndef NDEBUG
    params.above_edge_count[point_idx]++;
#endif
    return;
  }

  if (xsect_y > best_y) {
    auto deviant = scaling.UnscaleY(xsect_y) - scaling.UnscaleY(best_y);
    // continue search only if current primitive is not deviant too much
    if (best_y != init_best_y && deviant > params.early_term_deviant) {
      // FIXME(liang): early termination
      //      optixReportIntersection(0, 0);  // terminate ray
    }
#ifndef NDEBUG
    params.fail_update_count[point_idx]++;
#endif
    return;
  }

  if (xsect_y == best_y) {
    rayjoin::index_t best_e_eid = optixGetPayload_3();
    auto& best_e = params.base_map_edges[best_e_eid];
    auto current_e_slope = (double) e.a / e.b;
    auto best_e_slope = (double) best_e.a / best_e.b;
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
  optixSetPayload_3(eid);

#ifndef NDEBUG
  params.closer_count[point_idx]++;
#endif
}

extern "C" __global__ void __raygen__pip_custom() {
  float3 ray_dir = {0, 1, 0};
  const auto& query_points = params.query_points;
  const auto& scaling = params.scaling;

  for (unsigned int point_idx = OPTIX_TID_1D; point_idx < query_points.size();
       point_idx += OPTIX_TOTAL_THREADS_1D) {
    const auto& p = query_points[point_idx];
    const auto x = scaling.UnscaleX(p.x), y = scaling.UnscaleY(p.y);
    const auto fx = static_cast<float>(x), fy = static_cast<float>(y);
    const float tmin = 0;
    const float tmax = RAY_TMAX;
    auto best_y = std::numeric_limits<double>::infinity();
    static_assert(sizeof(best_y) == 8,
                  "Invalid internal coordinate type");  // current we use int64
    static_assert(sizeof(rayjoin::index_t) == 4, "Currently only support u32");
    uint2 best_y_storage;
    pack64(&best_y, best_y_storage.x, best_y_storage.y);
    rayjoin::index_t best_e_eid = std::numeric_limits<rayjoin::index_t>::max();
    float3 ray_origin;

    if (fy > y) {
      ray_origin.y = next_float_from_double(fy, -1, 1);
    } else {
      ray_origin.y = fy;
    }

    assert(ray_origin.y <= y);

    ray_origin.z = 0;

    //    if (fx == x) {

    ray_origin.x = fx;

    //    ray_origin.x = next_float_from_double(ray_origin.x, -1, 1);

    optixTrace(params.traversable, ray_origin, ray_dir,
               tmin,  // tmin
               tmax,  // tmax
               0,     // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               point_idx, best_y_storage.x, best_y_storage.y, best_e_eid);

    params.closest_eids[point_idx] = best_e_eid;
    //    } else {
    //      ray_origin.x = next_float_from_double(fx, -1, 1);
    //
    //      optixTrace(params.traversable, ray_origin, ray_dir,
    //                 tmin,  // tmin
    //                 tmax,  // tmax
    //                 0,     // rayTime
    //                 OptixVisibilityMask(255),
    //                 OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
    //                 SURFACE_RAY_TYPE,     // SBT offset
    //                 RAY_TYPE_COUNT,       // SBT stride
    //                 SURFACE_RAY_TYPE,     // missSBTIndex
    //                 point_idx, best_y_storage.x, best_y_storage.y,
    //                 best_e_eid_storage.x, best_e_eid_storage.y);
    //
    //      ray_origin.x = next_float_from_double(fx, 1, 1);
    //
    //      optixTrace(params.traversable, ray_origin, ray_dir,
    //                 tmin,  // tmin
    //                 tmax,  // tmax
    //                 0,     // rayTime
    //                 OptixVisibilityMask(255),
    //                 OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
    //                 SURFACE_RAY_TYPE,     // SBT offset
    //                 RAY_TYPE_COUNT,       // SBT stride
    //                 SURFACE_RAY_TYPE,     // missSBTIndex
    //                 point_idx, best_y_storage.x, best_y_storage.y,
    //                 best_e_eid_storage.x, best_e_eid_storage.y);
    //    }
  }
}
