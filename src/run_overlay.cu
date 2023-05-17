
#include <array>
#include <memory>

#include "app/overlay.h"
#include "app/rt_overlay.h"
#include "context.h"
#include "glog/logging.h"
#include "map/planar_graph.h"
#include "run_overlay.cuh"
#include "util/rational.h"
#include "util/stopwatch.h"
#include "util/timer.h"

namespace rayjoin {
template <typename CONTEXT_T, typename OVERLAY_IMPL_T>
void CheckResult(CONTEXT_T& ctx, OVERLAY_IMPL_T& overlay,
                 const OverlayConfig& config) {
  using xsect_t = dev::Intersection<typename CONTEXT_T::internal_coord_t>;
  using map_t = typename CONTEXT_T::map_t;
  MapOverlay<CONTEXT_T> cuda_grid(ctx, config.grid_size, config.xsect_factor);

  cuda_grid.Init();
  cuda_grid.AddMapsToGrid();
  cuda_grid.IntersectEdge(true);
  //  cuda_grid.ComputeOutputPolygons();
  {
    auto n_xsects_ans = cuda_grid.get_xsect_edges().size();
    auto n_xsects_res = overlay.get_xsect_edges_queue().size();
    int n_diff = abs((int) (n_xsects_ans - n_xsects_res));

    if (n_diff != 0) {
      LOG(ERROR) << "LSI "
                 << " xsects (Answer): " << n_xsects_ans
                 << " xsects (Result): " << n_xsects_res
                 << " Error rate: " << (double) n_diff / n_xsects_ans * 100
                 << " %";

      auto write_to = [](ArrayView<xsect_t> xsects, const char* path) {
        pinned_vector<xsect_t> h_xsects;
        h_xsects.resize(xsects.size());
        thrust::copy(thrust::device, xsects.data(),
                     xsects.data() + xsects.size(), h_xsects.begin());

        thrust::sort(h_xsects.begin(), h_xsects.end(),
                     [](const xsect_t& x1, const xsect_t& x2) {
                       return x1.eid[0] != x2.eid[0] ? x1.eid[0] < x2.eid[0]
                                                     : (x1.eid[1] < x2.eid[1]);
                     });

        std::ofstream ofs(path);

        for (auto& xsect : h_xsects) {
          ofs << xsect.eid[0] << " " << xsect.eid[1] << "\n";
        }

        ofs.close();
      };

      write_to(ArrayView<xsect_t>(cuda_grid.get_xsect_edges()),
               "/tmp/xsects.ans");
      write_to(overlay.get_xsect_edges_queue(), "/tmp/xsects.res");
    } else {
      LOG(INFO) << "LSI passed check";
    }
  }

  FOR2 {
    LOG(INFO) << "Checking point in polygon";

    overlay.LocateVerticesInOtherMap(im);
    cuda_grid.LocateVerticesInOtherMap(im);

    pinned_vector<index_t> closest_eids_ans(cuda_grid.get_closet_eids());
    pinned_vector<index_t> closest_eids_res(overlay.get_closet_eids());
    size_t n_diff = 0;
    size_t n_points = closest_eids_ans.size();

    CHECK_EQ(closest_eids_res.size(), n_points);

    auto query_map = ctx.get_map(im);
    auto base_map = ctx.get_map(1 - im);
    auto scaling = ctx.get_scaling();

    for (size_t point_idx = 0; point_idx < n_points; point_idx++) {
      auto closest_eid_ans = closest_eids_ans[point_idx];
      auto closest_eid_res = closest_eids_res[point_idx];
      // different eid does not mean wrong answer because there are two edges
      // having same coordinates but different eid
      if (closest_eid_res != closest_eid_ans) {
        auto not_hit = std::numeric_limits<index_t>::max();
        auto p = ctx.get_planar_graph(im)->points[point_idx];
        auto scaled_p = query_map->get_point(point_idx);
        bool diff = false;
        std::string ep_ans = "miss";
        std::string ep_res = "miss";
        std::string scaled_ep_ans = "miss";
        std::string scaled_ep_res = "miss";

        if (closest_eid_ans != not_hit) {
          ep_ans = base_map->EndpointsToString(closest_eid_ans, scaling);
          scaled_ep_ans = base_map->ScaledEndpointsToString(closest_eid_ans);
        }

        if (closest_eid_res != not_hit) {
          ep_res = base_map->EndpointsToString(closest_eid_res, scaling);
          scaled_ep_res = base_map->ScaledEndpointsToString(closest_eid_res);
        }

        if (scaled_ep_res != scaled_ep_ans) {
          diff = true;
        }

        if (diff && n_diff < 10) {
          printf("point %lu (%.8lf, %.8lf) ans %u, res %u %s %s\n", point_idx,
                 p.x, p.y, closest_eid_ans, closest_eid_res, ep_ans.c_str(),
                 ep_res.c_str());
          printf("scaled point %lu (%ld, %ld) ans %u, res %u %s %s\n",
                 point_idx, scaled_p.x, scaled_p.y, closest_eid_ans,
                 closest_eid_res, scaled_ep_ans.c_str(), scaled_ep_res.c_str());
        }

        if (diff) {
          n_diff++;
        }
      }
    }
    if (n_diff != 0) {
      LOG(ERROR) << "Map: " << im << " Total points: " << n_points
                 << " n diff: " << n_diff
                 << " Error rate: " << (double) n_diff / n_points * 100 << " %";
    } else {
      LOG(INFO) << "Map: " << im << " passed check";
    }
  }
}

void RunOverlay(const OverlayConfig& config) {
  using context_t = Context<coord_t, coefficient_t>;
  timer_start();

  timer_next("Read map 0");
  auto g1 = load_from<coord_t>(config.map1_path, config.serialize_prefix);
  timer_next("Read map 1");
  auto g2 = load_from<coord_t>(config.map2_path, config.serialize_prefix);

  timer_next("Create Context");
  context_t ctx({g1, g2});

  if (config.mode == "rt") {
    RTMapOverlay<context_t> overlay(ctx, config);

    timer_next("Init");
    overlay.Init();

    FOR2 {
      auto prefix = "Map " + std::to_string(im) + ": ";
      timer_next(prefix + "Build BVH");

      overlay.BuildBVH(im);
    };

    timer_next("Intersection edges");
    overlay.IntersectEdge();

    FOR2 {
      auto prefix = "Map " + std::to_string(im) + ": ";

      timer_next(prefix + "Locate vertices in other map");
      overlay.LocateVerticesInOtherMap(im);

      if (!config.profiling.empty()) {
        auto log_path =
            config.profiling + "/pip_query_map_" + std::to_string(im) + ".csv";
        LOG(INFO) << "Dumping PIP profiling results to " << log_path;
        overlay.DumpStatistics(log_path.c_str());
      }
    }

    timer_next("Dump Intersection");
    overlay.DumpIntersection();

    timer_next("Computer output polygons");
    overlay.ComputeOutputPolygons();

    if (config.check) {
      timer_next("Check result");
      CheckResult(ctx, overlay, config);
    }

    if (!config.output_path.empty()) {
      timer_next("Write to file");
      overlay.WriteResult(config.output_path.c_str());
    }

  } else if (config.mode == "grid") {
    MapOverlay<context_t> overlay(ctx, config.grid_size, config.xsect_factor);

    timer_next("Init");
    overlay.Init();

    timer_next("Add map to grid");
    overlay.AddMapsToGrid();

    timer_next("Intersect edges");
    overlay.IntersectEdge(config.lb);

    FOR2 {
      timer_next("Map " + std::to_string(im) +
                 ": Locate vertices in other map");
      overlay.LocateVerticesInOtherMap(im);
    }

    timer_next("Computer output polygons");
    overlay.ComputeOutputPolygons();

    if (!config.output_path.empty()) {
      timer_next("Write to file");
      overlay.WriteResult(config.output_path.c_str());
    }
  } else {
    LOG(FATAL) << "Illegal mode: " << config.mode;
  }
  timer_end();
}

}  // namespace rayjoin
