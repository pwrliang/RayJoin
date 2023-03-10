
#include <array>
#include <memory>

#include "app/overlay.h"
#include "app/rt_overlay.h"
#include "context.h"
#include "glog/logging.h"
#include "lbvh/lbvh_overlay.h"
#include "map/planar_graph.h"
#include "run_overlay.cuh"
#include "util/rational.h"
#include "util/stopwatch.h"
#include "util/timer.h"

namespace rayjoin {
template <typename CONTEXT_T>
void DebugPrint(CONTEXT_T& ctx) {
#ifndef NDEBUG
  auto print_edges = [](CONTEXT_T& ctx, int map_id,
                        const thrust::device_vector<index_t>& eids) {
    auto& stream = ctx.get_stream();
    auto d_map = ctx.get_map(map_id)->DeviceObject();
    auto scaling = ctx.get_scaling();
    printf("=========== Map %d Edges ===========\n", map_id);
    ForEach(
        stream, eids.size(),
        [=] __device__(size_t idx, ArrayView<index_t> eids) {
          auto eid = eids[idx];
          assert(eid < d_map.get_edges_num());
          const auto& e = d_map.get_edge(eid);
          auto& p1 = d_map.get_point(e.p1_idx);
          auto& p2 = d_map.get_point(e.p2_idx);
          auto x1 = scaling.UnscaleX(p1.x);
          auto y1 = scaling.UnscaleY(p1.y);
          auto x2 = scaling.UnscaleX(p2.x);
          auto y2 = scaling.UnscaleY(p2.y);

          printf("eid: %u (%lf, %lf) - (%lf, %lf)\n", (index_t) eid, x1, y1, x2,
                 y2);
        },
        ArrayView<index_t>(eids));
    stream.Sync();
  };
  {
    thrust::device_vector<index_t> eids;
    eids.push_back(93615);
    eids.push_back(226567);
    print_edges(ctx, 0, eids);
  }

  {
    thrust::device_vector<index_t> eids;
    eids.push_back(87230);
    eids.push_back(230119);
    print_edges(ctx, 1, eids);
  }
#endif
}

template <typename CONTEXT_T, typename OVERLAY_IMPL_T>
void CheckResult(CONTEXT_T& ctx, OVERLAY_IMPL_T& overlay,
                 const OverlayConfig& config) {
  using xsect_t = dev::Intersection<typename CONTEXT_T::internal_coord_t>;
  MapOverlay<CONTEXT_T> cuda_grid(ctx, config.grid_size, config.xsect_factor);

  cuda_grid.Init();
  cuda_grid.AddMapsToGrid();
  cuda_grid.IntersectEdge();
  cuda_grid.ComputeOutputPolygons();
  {
    auto total_n_xsects = cuda_grid.get_xsect_edges().size();
    auto rt_n_xsects = overlay.get_xsect_edges_queue().size();
    int n_diff = total_n_xsects - rt_n_xsects;

    if (n_diff != 0) {
      LOG(ERROR) << "LSI "
                 << " Total xsects: " << total_n_xsects << " n diff: " << n_diff
                 << " Error rate: " << (double) n_diff / total_n_xsects * 100
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
               "/tmp/xsects.grid");
      write_to(overlay.get_xsect_edges_queue(), "/tmp/xsects.rt");
      DebugPrint(ctx);
    } else {
      LOG(INFO) << "LSI passed check";
    }
  }
  FOR2 {
    LOG(INFO) << "Checking point in polygon";

    cuda_grid.LocateVerticesInOtherMap(im);

    pinned_vector<polygon_id_t> point_in_polygon(
        cuda_grid.get_point_in_polygon(im));
    pinned_vector<polygon_id_t> point_in_polygon_rt(
        overlay.get_point_in_polygon(im));
    size_t n_diff = 0;

    CHECK_EQ(point_in_polygon.size(), point_in_polygon_rt.size());

    for (size_t point_idx = 0; point_idx < point_in_polygon.size();
         point_idx++) {
      if (point_in_polygon[point_idx] != point_in_polygon_rt[point_idx]) {
        n_diff++;
        if (n_diff < 10) {
          auto p = ctx.get_planar_graph(im)->points[point_idx];

          printf("Diff! Map: %d point: %zu, (%lf, %lf), ans: %d, res: %d\n", im,
                 point_idx, (double) p.x, (double) p.y,
                 point_in_polygon[point_idx], point_in_polygon_rt[point_idx]);
        }
      }
    }
    if (n_diff != 0) {
      LOG(ERROR) << "Map: " << im
                 << " Total points: " << point_in_polygon.size()
                 << " n diff: " << n_diff << " Error rate: "
                 << (double) n_diff / point_in_polygon.size() * 100 << " %";
    } else {
      LOG(INFO) << "Map: " << im << " passed check";
    }
#if 0
    LOG(INFO) << "Checking intersections";

    thrust::host_vector<xsect_t> xsect(overlay.get_xsect_edges(im));
    thrust::host_vector<xsect_t> xsect_rt(overlay.get_xsect_edges(im));
    auto comparator = [=](const xsect_t& x1, const xsect_t& x2) {
      if (x1.x != x2.x) {
        return x1.x < x2.x;
      }
      //      if (x1.y != x2.y) {
      return x1.y < x2.y;
      //      }
      //      if (x1.eid[0] != x2.eid[0]) {
      //        return x1.eid[0] < x2.eid[0];
      //      }
      //      return x1.eid[1] < x2.eid[1];
    };

    std::sort(xsect.begin(), xsect.end(), comparator);
    std::sort(xsect_rt.begin(), xsect_rt.end(), comparator);

    CHECK_EQ(xsect.size(), xsect_rt.size());
    for (size_t xsect_idx = 0; xsect_idx < xsect.size(); xsect_idx++) {
      auto& xsect_ans = xsect[xsect_idx];
      auto& xsect_res = xsect_rt[xsect_idx];

      //      if (xsect_ans.eid[0] != xsect_res.eid[0] ||
      //          xsect_ans.eid[1] != xsect_res.eid[1]) {
      //        n_diff++;
      //      }

      if (xsect_ans.x != xsect_res.x || xsect_ans.y != xsect_res.y) {
        n_diff++;
      }
      if (xsect_ans.mid_point_polygon_id != xsect_res.mid_point_polygon_id) {
        LOG(INFO) << xsect_ans.mid_point_polygon_id << " vs "
                  << xsect_res.mid_point_polygon_id << " diff: "
                  << xsect_ans.mid_point_polygon_id -
                         xsect_res.mid_point_polygon_id;

        n_diff++;
      }
    }

    if (n_diff != 0) {
      LOG(FATAL) << "Map: " << im << " Total xsects: " << xsect.size()
                 << " n diff: " << n_diff;
    } else {
      LOG(INFO) << "Map: " << im << " passed check";
    }
#endif
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
    overlay.Init(config.exec_root);

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
    };

    //    timer_next("Dump Intersection");
    //    overlay.DumpIntersection();
    //
    //    timer_next("Print statistics");
    //    FOR2 { overlay.RTStatistics(im); };

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

    //    timer_next("Debug Print");
    //    DebugPrint(ctx);
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
