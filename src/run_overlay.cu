
#include <array>
#include <memory>

#include "app/map_overlay_grid.h"
#include "app/map_overlay_lbvh.h"
#include "app/map_overlay_rt.h"
#include "context.h"
#include "glog/logging.h"
#include "map/planar_graph.h"
#include "run_overlay.cuh"
#include "util/rational.h"
#include "util/stopwatch.h"
#include "util/timer.h"

namespace rayjoin {
template <typename CONTEXT_T, typename OVERLAY_IMPL_T>
void CheckResult(CONTEXT_T& ctx, std::shared_ptr<OVERLAY_IMPL_T> overlay,
                 const OverlayConfig& config) {
  using xsect_t = dev::Intersection<typename CONTEXT_T::internal_coord_t>;
  using map_t = typename CONTEXT_T::map_t;
  MapOverlayGrid<CONTEXT_T> cuda_grid(ctx);

  QueryConfigGrid query_config;

  LOG(INFO) << "Checking LSI Results";
  query_config.xsect_factor = config.xsect_factor;
  query_config.grid_size = config.grid_size;
  cuda_grid.set_config(query_config);
  cuda_grid.Init();
  cuda_grid.BuildIndex();
  cuda_grid.IntersectEdge(0);
  {
    auto xsects_ans = cuda_grid.get_xsect_edges();
    auto xsects_res = overlay->get_xsect_edges();
    auto n_xsects_ans = xsects_ans.size();
    auto n_xsects_res = xsects_res.size();
    auto write_to = [](const char* path, thrust::host_vector<xsect_t>& xsects) {
      thrust::sort(xsects.begin(), xsects.end(),
                   [](const xsect_t& a, const xsect_t& b) {
                     if (a.eid[0] != b.eid[0]) {
                       return a.eid[0] < b.eid[0];
                     }
                     return a.eid[1] < b.eid[1];
                   });

      std::ofstream ofs(path);
      for (auto& xsect : xsects) {
        ofs << xsect.eid[0] << " " << xsect.eid[1] << "\n";
      }
      ofs.close();
    };

    // it is possible that rt finds more xsects than the grid due to the
    // numerical issue of the grid
    if (n_xsects_ans != n_xsects_res) {
      auto n_diff = abs((int64_t) n_xsects_ans - (int64_t) n_xsects_res);

      LOG(ERROR) << "LSI "
                 << " xsects (Answer): " << n_xsects_ans
                 << " xsects (Result): " << n_xsects_res
                 << " False negative rate: "
                 << (double) n_diff / n_xsects_ans * 100 << " %";
      write_to("/tmp/xsects_answer.txt", xsects_ans);
      write_to("/tmp/xsects_result.txt", xsects_res);
    } else {
      LOG(INFO) << "LSI passed check";
    }
  }

  FOR2 {
    LOG(INFO) << "Checking point in polygon";

    cuda_grid.LocateVerticesInOtherMap(im);

    auto closest_eids_ans = cuda_grid.get_closet_eids(im);
    auto closest_eids_res = overlay->get_closet_eids(im);
    size_t n_diff = 0;
    size_t n_points = closest_eids_ans.size();

    CHECK_EQ(closest_eids_res.size(), n_points);

    auto query_map = ctx.get_map(im);
    auto base_map = ctx.get_map(1 - im);
    auto scaling = ctx.get_scaling();

    query_map->D2H();
    base_map->D2H();

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
      LOG(INFO) << "Map: " << im << " PIP passed check";
    }
  }
}

void RunOverlay(const OverlayConfig& config) {
  using context_t = Context<coord_t, coefficient_t>;
  timer_start();

  timer_next("Read map 0");
  LOG(INFO) << "Reading map 0 from " << config.map1_path;
  auto g1 = load_from<coord_t>(config.map1_path, config.serialize_prefix);

  timer_next("Read map 1");
  LOG(INFO) << "Reading map 1 from " << config.map2_path;
  auto g2 = load_from<coord_t>(config.map2_path, config.serialize_prefix);

  timer_next("Create App");
  context_t ctx({g1, g2});
  std::shared_ptr<MapOverlay<context_t>> overlay;

  if (config.mode == "rt") {
    auto overlay_rt = std::make_shared<MapOverlayRT<context_t>>(ctx);
    QueryConfigRT query_config;

    query_config.profile = config.profile;
    query_config.fau = config.fau;
    query_config.xsect_factor = config.xsect_factor;
    query_config.ag = config.ag;
    query_config.ag_iter = config.ag_iter;
    query_config.win = config.win;
    query_config.enlarge = config.enlarge;

    overlay_rt->set_config(query_config);
    overlay = overlay_rt;
  } else if (config.mode == "grid") {
    auto overlay_grid = std::make_shared<MapOverlayGrid<context_t>>(ctx);
    QueryConfigGrid query_config;

    query_config.grid_size = config.grid_size;
    query_config.profile = config.profile;
    query_config.xsect_factor = config.xsect_factor;

    overlay_grid->set_config(query_config);
    overlay = overlay_grid;
  } else if (config.mode == "lbvh") {
    auto overlay_lbvh = std::make_shared<MapOverlayLBVH<context_t>>(ctx);
    QueryConfigLBVH query_config;

    query_config.profile = config.profile;
    query_config.xsect_factor = config.xsect_factor;

    overlay_lbvh->set_config(query_config);
    overlay = overlay_lbvh;
  } else {
    LOG(FATAL) << "Illegal mode: " << config.mode;
  }

  timer_next("Load Data");
  ctx.LoadToDevice();

  timer_next("Init");
  overlay->Init();

  timer_next("Build Index");
  overlay->BuildIndex();

  timer_next("Intersection edges");
  overlay->IntersectEdge(0);

  FOR2 {
    auto prefix = "Map " + std::to_string(im) + ": ";

    timer_next(prefix + "Locate vertices in other map");
    overlay->LocateVerticesInOtherMap(im);
  }

  timer_next("Computer output polygons");
  overlay->ComputeOutputPolygons();

  if (config.check) {
    timer_next("Check result");
    CheckResult(ctx, overlay, config);
  }

  if (!config.output_path.empty()) {
    timer_next("Write to file");
    overlay->WriteResult(config.output_path.c_str());
  }
  timer_end();
}

}  // namespace rayjoin
