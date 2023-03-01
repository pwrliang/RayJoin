#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_device.h>

#include "algo/launch_parameters.h"
#include "algo/lsi.h"
#include "config.h"
#include "util/helpers.h"
#include "util/int128_intrinsics.h"
#include "util/util.h"
#include "util/vec_math.h"

enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };

extern "C" __constant__ rayjoin::LaunchParamsLSI params;

extern "C" __global__ void __intersection__lsi() {
  using xsect_t = typename rayjoin::LaunchParamsLSI::xsect_t;
  using edge_t = typename rayjoin::LaunchParamsLSI::edge_t;
  using point_t = typename rayjoin::LaunchParamsLSI::point_t;
  auto base_eid = optixGetPrimitiveIndex();
  auto query_eid = optixGetPayload_0();
  const auto& base_e = params.base_edges[base_eid];
  const auto& query_e = params.query_edges[query_eid];

  const auto& base_e_p1 = params.base_points[base_e.p1_idx];
  const auto& base_e_p2 = params.base_points[base_e.p2_idx];
  const auto& query_e_p1 = params.query_points[query_e.p1_idx];
  const auto& query_e_p2 = params.query_points[query_e.p2_idx];

  if (rayjoin::dev::intersect_test<edge_t, edge_t, point_t, double>(
          base_e, base_e_p1, base_e_p2, query_e, query_e_p1, query_e_p2)) {
    xsect_t xsect;

    xsect.eid[0] = base_eid;
    xsect.eid[1] = query_eid;
    params.xsects.Append(xsect);
  }
}

extern "C" __global__ void __raygen__lsi() {
  const auto& scaling = params.scaling;
  const auto& edges = params.query_edges;
  float ray_z = 0;

  for (unsigned int eid = OPTIX_TID_1D; eid < edges.size();
       eid += OPTIX_TOTAL_THREADS_1D) {
    auto& e = edges[eid];
    auto p1 = params.query_points[e.p1_idx];
    auto p2 = params.query_points[e.p2_idx];
    if (p1.x > p2.x) {
      SWAP(p1, p2);
    }
    double2 p1_coord{scaling.UnscaleX(p1.x), scaling.UnscaleY(p1.y)};
    double2 p2_coord{scaling.UnscaleX(p2.x), scaling.UnscaleY(p2.y)};
    double2 ext_p1 = p1_coord, ext_p2 = p2_coord;

    float3 ray_origin = {(float) ext_p1.x, (float) ext_p1.y, ray_z};
    float3 ray_dir = {(float) (ext_p2.x - ext_p1.x),
                      (float) (ext_p2.y - ext_p1.y), ray_z};

    float tmin = 0;
    float tmax = 1;
    float3 shifted_origin = ray_origin - RAY_OFFSET_T * ray_dir;

    optixTrace(params.traversable, ray_origin, ray_dir,
               tmin,                 // tmin
               tmax + RAY_OFFSET_T,  // tmax
               0,                    // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               eid);
  }
}
