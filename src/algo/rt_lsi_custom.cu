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

extern "C" __global__ void __intersection__lsi_custom() {
  using xsect_t = typename rayjoin::LaunchParamsLSI::xsect_t;
  using edge_t = typename rayjoin::LaunchParamsLSI::edge_t;
  using point_t = typename rayjoin::LaunchParamsLSI::point_t;
  auto prim_idx = optixGetPrimitiveIndex();
  auto query_eid = optixGetPayload_0();
  auto query_map_id = params.query_map_id;
  const auto& query_e = params.query_edges[query_eid];
  const auto& query_e_p1 = params.query_points[query_e.p1_idx];
  const auto& query_e_p2 = params.query_points[query_e.p2_idx];
  auto begin_eid = params.eid_range[prim_idx].first;
  auto end_eid = params.eid_range[prim_idx].second;

  for (auto base_eid = begin_eid; base_eid < end_eid; base_eid++) {
    const auto& base_e = params.base_edges[base_eid];
    const auto& base_e_p1 = params.base_points[base_e.p1_idx];
    const auto& base_e_p2 = params.base_points[base_e.p2_idx];

    if (rayjoin::dev::intersect_test<edge_t, edge_t, point_t,
                                     rayjoin::coefficient_t>(
            query_e, query_e_p1, query_e_p2, base_e, base_e_p1, base_e_p2)) {
      xsect_t xsect;
      xsect.eid[query_map_id] = query_eid;
      xsect.eid[1 - query_map_id] = base_eid;
      params.xsects.Append(xsect);
    }
  }
#ifndef NDEBUG
  atomicAdd(params.n_tests, end_eid - begin_eid);
#endif
}

extern "C" __global__ void __raygen__lsi_custom() {
  using coefficient_t = rayjoin::coefficient_t;
  const auto& scaling = params.scaling;
  const auto& edges = params.query_edges;
  float tmin = 0;
  float tmax = 1;

  for (unsigned int eid = OPTIX_TID_1D; eid < edges.size();
       eid += OPTIX_TOTAL_THREADS_1D) {
    auto& e = edges[eid];
    auto p1 = params.query_points[e.p1_idx];
    auto p2 = params.query_points[e.p2_idx];
    double x1, x2, y1, y2;

    if (e.b == 0) {
      assert(p1.x == p2.x);
      if (p1.y > p2.y) {
        SWAP(p1, p2);
      }
      // y1 < y2
      x1 = x2 = scaling.UnscaleX(p1.x);
      y1 = scaling.UnscaleY(p1.y);
      y2 = scaling.UnscaleY(p2.y);
    } else {
      assert(p1.x != p2.x);
      if (p1.x > p2.x) {
        SWAP(p1, p2);
      }

      x1 = scaling.UnscaleX(p1.x);
      y1 = scaling.UnscaleY(p1.y);
      x2 = scaling.UnscaleX(p2.x);
      y2 = scaling.UnscaleY(p2.y);
    }

    float3 ray_origin = {x1, y1, 0};
    float3 ray_dir = {x2 - x1, y2 - y1, 0};

    optixTrace(params.traversable, ray_origin, ray_dir,
               tmin,  // tmin
               tmax,  // tmax
               0,     // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,  // OPTIX_RAY_FLAG_NONE,
               SURFACE_RAY_TYPE,     // SBT offset
               RAY_TYPE_COUNT,       // SBT stride
               SURFACE_RAY_TYPE,     // missSBTIndex
               eid);
  }
}
