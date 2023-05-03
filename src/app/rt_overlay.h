#ifndef RAYJOIN_APP_RT_OVERLAY_H
#define RAYJOIN_APP_RT_OVERLAY_H
#include <iomanip>
#include <random>

#include "app/output_chain.h"
#include "app/overlay_config.h"
#include "app/pip_rt.h"
#include "grid/uniform_grid.h"
#include "rt/rt_engine.h"
#include "util/bitset.h"
#include "util/cta_scheduler.h"
#include "util/helper_mortonCode.h"
#include "util/helpers.h"
#include "util/stopwatch.h"
#include "util/util.h"

namespace rayjoin {

template <typename CONTEXT_T>
class RTMapOverlay {
  using coord_t = typename CONTEXT_T::coord_t;
  using internal_coord_t = typename CONTEXT_T::internal_coord_t;
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using xsect_t = dev::Intersection<internal_coord_t>;

 public:
  explicit RTMapOverlay(CONTEXT_T& ctx, const OverlayConfig& config)
      : ctx_(ctx),
        config_(config),
        rt_engine_(std::make_shared<RTEngine>()),
        pip_(ctx, rt_engine_) {
    size_t total_n_edges = 0;

    FOR2 {
      auto n_edges = ctx.get_map(im)->get_edges_num();

      total_n_edges += n_edges;
      max_n_xsects_ += n_edges * config_.xsect_factor;
    }

    LOG(INFO) << "Total edges: " << total_n_edges
              << " Xsect Factor: " << config_.xsect_factor
              << " Max xsects: " << max_n_xsects_;
  }

  void Init() {
    auto& stream = ctx_.get_stream();
    size_t required_space = 0;
    auto exec_root = ctx_.get_exec_root();
    RTConfig rt_config = get_default_rt_config(exec_root);

    required_space += SIZEOF_ELEM(xsect_edges_) * max_n_xsects_ * 3;
    FOR2 {
      auto points_num = ctx_.get_map(im)->get_points_num();
      required_space += SIZEOF_ELEM(point_in_polygon_[im]) * points_num;
      required_space += SIZEOF_ELEM(xsect_edges_sorted_[im]);
#ifndef NDEBUG
      hit_count_[im].resize(points_num, 0);
      closer_count_[im].resize(points_num, 0);
      above_edge_count_[im].resize(points_num, 0);
      fail_update_count_[im].resize(points_num, 0);
#endif
    }
    LOG(INFO) << "Required Space: " << required_space / 1024 / 1024 << " MB";

    xsect_edges_.Init(max_n_xsects_ * 3);

    size_t max_ne = 0;
    FOR2 {
      auto map = ctx_.get_map(im);
      auto points_num = map->get_points_num();
      auto ne = map->get_edges_num();

      point_in_polygon_[im].resize(points_num);
      max_ne = std::max(max_ne, ne);
    }

    rt_engine_->Init(rt_config);
    pip_.Init(max_ne);

    stream.Sync();
  }

  void BuildBVH(int map_id) {
    auto& stream = ctx_.get_stream();
    const auto& scaling = ctx_.get_scaling();
    auto d_map = ctx_.get_map(map_id)->DeviceObject();
    auto ne = d_map.get_edges_num();
    auto rounding_iter = config_.rounding_iter;

    if (config_.use_triangle) {
      auto bb = ctx_.get_bounding_box();

      triangle_points_.resize(d_map.get_edges_num() * 3);

      ArrayView<float3> d_triangle_points(triangle_points_);

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

          y1 =
              next_float_from_double(scaling.UnscaleY(p1.y), -1, rounding_iter);
          y2 = next_float_from_double(scaling.UnscaleY(p2.y), 1, rounding_iter);
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

          x1 =
              next_float_from_double(scaling.UnscaleX(p1.x), -1, rounding_iter);
          y1 = scaling.UnscaleY(a * scaling.ScaleX(x1) + b);

          x2 = next_float_from_double(scaling.UnscaleX(p2.x), 1, rounding_iter);
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
      traverse_handles_[map_id] =
          rt_engine_->BuildAccelTriangles(stream, d_triangle_points);
    } else {
      aabbs_.resize(ne);

      ArrayView<OptixAabb> d_aabbs(aabbs_);

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

        aabb.minX = next_float_from_double(min(x1, x2), -1, rounding_iter);
        aabb.maxX = next_float_from_double(max(x1, x2), 1, rounding_iter);
        aabb.minY = next_float_from_double(min(y1, y2), -1, rounding_iter);
        aabb.maxY = next_float_from_double(max(y1, y2), 1, rounding_iter);
        aabb.minZ = -PRIMITIVE_HEIGHT / 2;
        aabb.maxZ = PRIMITIVE_HEIGHT / 2;
      });
      traverse_handles_[map_id] = rt_engine_->BuildAccelCustom(stream, d_aabbs);
    }

    stream.Sync();
    if (config_.fau) {
      if (config_.use_triangle) {
        triangle_points_.resize(0);
        triangle_points_.shrink_to_fit();
      } else {
        aabbs_.resize(0);
        aabbs_.shrink_to_fit();
      }
    }
  }

  void IntersectEdge() {
    auto& stream = ctx_.get_stream();
    auto& scaling = ctx_.get_scaling();
    int query_map_id = 1;
    int base_map_id = 1 - query_map_id;
    auto d_base_map = ctx_.get_map(base_map_id)->DeviceObject(),
         d_query_map = ctx_.get_map(query_map_id)->DeviceObject();
    auto module_id = config_.use_triangle
                         ? ModuleIdentifier::MODULE_ID_LSI
                         : ModuleIdentifier::MODULE_ID_LSI_CUSTOM;
    LaunchParamsLSI params;

    params.query_map_id = query_map_id;
    params.scaling = scaling;
    params.base_edges = d_base_map.get_edges().data();
    params.base_points = d_base_map.get_points().data();
    params.query_edges = d_query_map.get_edges();
    params.query_points = d_query_map.get_points().data();
    params.traversable = traverse_handles_[base_map_id];
    params.rounding_iter = config_.rounding_iter;
    params.xsects = xsect_edges_.DeviceObject();

    xsect_edges_.Clear(stream);

    rt_engine_->CopyLaunchParams(stream, params);

    rt_engine_->Render(stream, module_id,
                       dim3{(unsigned int) d_query_map.get_edges_num(), 1, 1});

    stream.Sync();

    size_t n_xsects = xsect_edges_.size(stream);
    ArrayView<xsect_t> d_xsects(xsect_edges_.data(), n_xsects);

    ForEach(stream, n_xsects, [=] __device__(size_t idx) mutable {
      auto& xsect = d_xsects[idx];
      const auto& query_e = d_query_map.get_edge(xsect.eid[query_map_id]);
      const auto& query_e_p1 = d_query_map.get_point(query_e.p1_idx);
      const auto& query_e_p2 = d_query_map.get_point(query_e.p2_idx);

      const auto& base_e = d_base_map.get_edge(xsect.eid[base_map_id]);
      const auto& base_e_p1 = d_base_map.get_point(base_e.p1_idx);
      const auto& base_e_p2 = d_base_map.get_point(base_e.p2_idx);

      auto denom = (coefficient_t) query_e.a * base_e.b -
                   (coefficient_t) base_e.a * query_e.b;
      auto numx = (coefficient_t) base_e.c * query_e.b -
                  (coefficient_t) query_e.c * base_e.b;
      auto numy = (coefficient_t) base_e.a * query_e.c -
                  (coefficient_t) query_e.a * base_e.c;

      tcb::rational<__int128> xsect_x(numx, denom);
      tcb::rational<__int128> xsect_y(numy, denom);

      auto t = MIN4(query_e_p1.x, query_e_p2.x, base_e_p1.x, base_e_p2.x);
      if (xsect_x < t) {
        xsect_x = t;
      }

      t = MAX4(query_e_p1.x, query_e_p2.x, base_e_p1.x, base_e_p2.x);
      if (xsect_x > t) {
        xsect_x = t;
      }

      t = MIN4(query_e_p1.y, query_e_p2.y, base_e_p1.y, base_e_p2.y);
      if (xsect_y < t) {
        xsect_y = t;
      }
      t = MAX4(query_e_p1.y, query_e_p2.y, base_e_p1.y, base_e_p2.y);
      if (xsect_y > t) {
        xsect_y = t;
      }

      xsect.x = xsect_x;
      xsect.y = xsect_y;
    });

    stream.Sync();

    LOG(INFO) << "RT Xsects: " << n_xsects;
  }

  void DumpIntersection() {
    auto& stream = ctx_.get_stream();
    auto scaling = ctx_.get_scaling();
    pinned_vector<xsect_t> xsects;

    xsects.resize(xsect_edges_.size());

    thrust::copy(thrust::cuda::par.on(stream.cuda_stream()),
                 xsect_edges_.data(), xsect_edges_.data() + xsect_edges_.size(),
                 xsects.begin());
    stream.Sync();

    auto sort = [](pinned_vector<xsect_t>& xsects) {
      thrust::sort(xsects.begin(), xsects.end(),
                   [](const xsect_t& a, const xsect_t& b) {
                     if (a.eid[0] != b.eid[0]) {
                       return a.eid[0] < b.eid[0];
                     }
                     return a.eid[1] < b.eid[1];
                   });
    };

    sort(xsects);

    auto out_to_file = [scaling](const char* file,
                                 pinned_vector<xsect_t>& xsects,
                                 bool coord = false) {
      std::ofstream ofs(file);
      for (auto& xsect : xsects) {
        if (coord) {
          ofs << xsect.eid[0] << " " << xsect.eid[1] << std::fixed
              << std::setprecision(8) << " (" << scaling.UnscaleX(xsect.x)
              << ", " << scaling.UnscaleY(xsect.y)
              << ")"
                 "\n";
        } else {
          ofs << xsect.eid[0] << " " << xsect.eid[1] << "\n";
        }
      }
      ofs.close();
    };
    out_to_file("/tmp/xsect_rt.txt", xsects);
    out_to_file("/tmp/xsect_rt_coord.txt", xsects, true);
  }

  void LocateVerticesInOtherMap(int query_map_id) {
    Stream& stream = ctx_.get_stream();
    auto base_map_id = 1 - query_map_id;
    auto d_base_map = ctx_.get_map(base_map_id)->DeviceObject();
    auto d_query_map = ctx_.get_map(query_map_id)->DeviceObject();
    auto d_points = d_query_map.get_points();
    auto query_config = get_rt_query_config(base_map_id);

    pip_.set_query_config(query_config);
    pip_.Query(stream, base_map_id, d_points);

    auto& closest_eid = pip_.get_closest_eids();

    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      closest_eid.begin(), closest_eid.end(),
                      point_in_polygon_[query_map_id].begin(),
                      [=] __device__(index_t eid) {
                        // point is not in polygon
                        if (eid == std::numeric_limits<index_t>::max()) {
                          return EXTERIOR_FACE_ID;
                        }

                        const auto& e = d_base_map.get_edge(eid);

                        return d_base_map.get_face_id(e);
                      });
    stream.Sync();
  }

  void RTStatistics(int src_map_id) {
#ifndef NDEBUG
    pinned_vector<uint32_t> hit_count = hit_count_[src_map_id];
    pinned_vector<uint32_t> closer_count = closer_count_[src_map_id];
    pinned_vector<uint32_t> above_edge_count = above_edge_count_[src_map_id];
    pinned_vector<uint32_t> fail_update_count = fail_update_count_[src_map_id];

    uint32_t max_hit_count = 0;
    uint32_t max_closer_count = 0;
    uint32_t max_above_edge_count = 0;
    uint32_t max_fail_update_count = 0;
    size_t total_hit_count = 0;
    size_t total_closer_count = 0;
    size_t total_above_edge_count = 0;
    size_t total_fail_update_count = 0;
    index_t pid_max_hit_count = 0;
    index_t pid_max_closer_count = 0;
    index_t pid_max_above_edge_count = 0;
    index_t pid_max_fail_update_count = 0;

    for (size_t point_idx = 0; point_idx < hit_count.size(); point_idx++) {
      if (hit_count[point_idx] > max_hit_count) {
        max_hit_count = hit_count[point_idx];
        pid_max_hit_count = point_idx;
      }
      if (closer_count[point_idx] > max_closer_count) {
        max_closer_count = closer_count[point_idx];
        pid_max_closer_count = point_idx;
      }
      if (above_edge_count[point_idx] > max_above_edge_count) {
        max_above_edge_count = above_edge_count[point_idx];
        pid_max_above_edge_count = point_idx;
      }
      if (fail_update_count[point_idx] > max_fail_update_count) {
        max_fail_update_count = fail_update_count[point_idx];
        pid_max_fail_update_count = point_idx;
      }

      total_hit_count += hit_count[point_idx];
      total_closer_count += closer_count[point_idx];
      total_above_edge_count += above_edge_count[point_idx];
      total_fail_update_count += fail_update_count[point_idx];
    }

    LOG(INFO) << "--------- MAP: " << src_map_id << " ---------";

    LOG(INFO) << "Max hit count: " << max_hit_count
              << " pid: " << pid_max_hit_count
              << " Avg hit count: " << total_hit_count / hit_count.size();
    LOG(INFO) << "Max closer count: " << max_closer_count
              << " pid: " << pid_max_closer_count
              << " Avg closer count: " << total_closer_count / hit_count.size();
    LOG(INFO) << "Max Above edge count: " << max_above_edge_count
              << " pid: " << pid_max_above_edge_count
              << " Avg Above edge count: "
              << total_above_edge_count / hit_count.size();
    LOG(INFO) << "Max fail update count: " << max_fail_update_count
              << " pid: " << pid_max_fail_update_count
              << " Avg fail update count: "
              << total_fail_update_count / hit_count.size();
#endif
  }

  void ComputeOutputPolygons() {
    using point_t = typename CONTEXT_T::map_t::point_t;
    auto& stream = ctx_.get_stream();
    size_t n_xsects = xsect_edges_.size(stream);
    const auto& scaling = ctx_.get_scaling();

    FOR2 {
      // TODO: Move them out out the loop
      thrust::device_vector<index_t> unique_eids;
      thrust::device_vector<uint32_t> n_xsects_per_edge;
      thrust::device_vector<uint32_t> xsect_index;
      thrust::device_vector<point_t> mid_points;

      auto& xsect_edges_sorted = xsect_edges_sorted_[im];
      // Sort by eid1, eid2 respectively, so we can do binary search
      xsect_edges_sorted.resize(n_xsects);
      thrust::copy(thrust::cuda::par.on(stream.cuda_stream()),
                   xsect_edges_.data(), xsect_edges_.data() + n_xsects,
                   xsect_edges_sorted.begin());
      thrust::sort(
          thrust::cuda::par.on(stream.cuda_stream()),
          xsect_edges_sorted.begin(), xsect_edges_sorted.end(),
          [=] __device__(const xsect_t& xsect1, const xsect_t& xsect2) {
            return xsect1.eid[im] < xsect2.eid[im];
          });

      ArrayView<xsect_t> d_xsect_edges_sorted(xsect_edges_sorted);
      auto query_map_id = im, base_map_id = 1 - im;
      auto d_query_map = ctx_.get_map(query_map_id)->DeviceObject();
      auto d_base_map = ctx_.get_map(base_map_id)->DeviceObject();

      unique_eids.resize(n_xsects);

      // get eids of current map
      thrust::transform(
          thrust::cuda::par.on(stream.cuda_stream()),
          xsect_edges_sorted.begin(), xsect_edges_sorted.end(),
          unique_eids.begin(),
          [=] __device__(const xsect_t& xsect) { return xsect.eid[im]; });

      auto end = thrust::unique(thrust::cuda::par.on(stream.cuda_stream()),
                                unique_eids.begin(), unique_eids.end());

      unique_eids.resize(end - unique_eids.begin());
      n_xsects_per_edge.resize(unique_eids.size());
      xsect_index.resize(unique_eids.size() + 1, 0);

      ArrayView<uint32_t> d_n_xsects_per_edge(n_xsects_per_edge);
      // find intersections for each edge
      thrust::transform(
          thrust::cuda::par.on(stream.cuda_stream()), unique_eids.begin(),
          unique_eids.end(), n_xsects_per_edge.begin(),
          [=] __device__(index_t eid) {
            // Assuming points are organized respect to geometric space
            xsect_t dummy_xsect;
            dummy_xsect.eid[im] = eid;

            // get eid intersects which edges in other map
            auto it = thrust::equal_range(
                thrust::seq, d_xsect_edges_sorted.begin(),
                d_xsect_edges_sorted.end(), dummy_xsect,
                [=] __device__(const xsect_t& xsect1, const xsect_t& xsect2) {
                  return xsect1.eid[im] < xsect2.eid[im];
                });
            return thrust::distance(it.first, it.second);
          });

      thrust::inclusive_scan(thrust::cuda::par.on(stream.cuda_stream()),
                             n_xsects_per_edge.begin(), n_xsects_per_edge.end(),
                             xsect_index.begin() + 1);
      stream.Sync();
      // n intersection points have n-1 mid points
      uint32_t n_mid_points =
          xsect_index[xsect_index.size() - 1] - unique_eids.size();
      // mid-points of intersections from query map
      mid_points.resize(n_mid_points);

      ArrayView<uint32_t> d_xsect_index(xsect_index);
      ArrayView<point_t> d_mid_points(mid_points);
      ArrayView<index_t> d_unique_eids(unique_eids);

      // collect all mid points
      ForEach(stream, d_unique_eids.size(), [=] __device__(size_t idx) mutable {
        auto eid = d_unique_eids[idx];
        auto begin = d_xsect_index[idx];
        auto end = d_xsect_index[idx + 1];
        auto n_xsect = end - begin;

        if (n_xsect > 1) {
          const auto& e = d_query_map.get_edge(eid);
          const auto& p1 = d_query_map.get_point(e.p1_idx);
          auto* curr_xsects = d_xsect_edges_sorted.data() + begin;

          thrust::sort(
              thrust::seq, curr_xsects, d_xsect_edges_sorted.data() + end,
              [=](const xsect_t& xsect1, const xsect_t& xsect2) {
                // fixme: cast __128 directly
                auto d1 = SQ(xsect1.x - tcb::rational<__int128>(p1.x)) +
                          SQ(xsect1.y - tcb::rational<__int128>(p1.y));
                auto d2 = SQ(xsect2.x - tcb::rational<__int128>(p1.x)) +
                          SQ(xsect2.y - tcb::rational<__int128>(p1.y));
                return d1 < d2;
              });

          for (int xsect_idx = 0; xsect_idx < n_xsect - 1; xsect_idx++) {
            xsect_t& xsect1 = *(curr_xsects + xsect_idx);
            xsect_t& xsect2 = *(curr_xsects + xsect_idx + 1);
            tcb::rational<__int128> x1 = xsect1.x, y1 = xsect1.y;
            tcb::rational<__int128> x2 = xsect2.x, y2 = xsect2.y;
            dev::ExactPoint<internal_coord_t> mid_p(x1 + (x2 - x1) / 2,
                                                    y1 + (y2 - y1) / 2);

            assert(begin + xsect_idx - idx < d_mid_points.size());

            d_mid_points[begin + xsect_idx - idx] = {mid_p.x, mid_p.y};
          }
        }
      });

      stream.Sync();

      auto query_config = get_rt_query_config(base_map_id);

      pip_.set_query_config(query_config);
      pip_.Query(stream, base_map_id, d_mid_points);

      stream.Sync();

      ArrayView<index_t> d_mid_point_closest_eid(pip_.get_closest_eids());

      // fill in polygon id for mid-points
      ForEach(stream, d_unique_eids.size(), [=] __device__(size_t idx) mutable {
        auto begin = d_xsect_index[idx];
        auto end = d_xsect_index[idx + 1];
        auto n_xsect = end - begin;

        if (n_xsect > 1) {
          auto* curr_xsects = d_xsect_edges_sorted.data() + begin;

          for (int xsect_idx = 0; xsect_idx < n_xsect - 1; xsect_idx++) {
            xsect_t& xsect1 = *(curr_xsects + xsect_idx);
            auto eid = d_mid_point_closest_eid[begin + xsect_idx - idx];
            polygon_id_t ipol = EXTERIOR_FACE_ID;

            if (eid != std::numeric_limits<index_t>::max()) {
              const auto& e = d_base_map.get_edge(eid);
              ipol = d_base_map.get_face_id(e);
            }

            xsect1.mid_point_polygon_id = ipol;
          }
        }
      });
      stream.Sync();
    }
  }

  void WriteResult(const char* path) {
    WriteOutputChain(ctx_, xsect_edges_sorted_, point_in_polygon_, path);
  }

  void PrintMemoryUsage() {
    size_t bytes = 0;

    bytes += COUNT_CONTAINER_BYTES(xsect_edges_);
    FOR2 {
      bytes += COUNT_CONTAINER_BYTES(xsect_edges_sorted_[im]);
      bytes += COUNT_CONTAINER_BYTES(point_in_polygon_[im]);
    }

    LOG(INFO) << "Occupied Memory: " << bytes / 1024 / 1024 << " MB";
  }

  const thrust::device_vector<xsect_t>& get_xsect_edges(int im) const {
    return xsect_edges_sorted_[im];
  }

  const thrust::device_vector<index_t>& get_closet_eids() const {
    return pip_.get_closest_eids();
  }

  ArrayView<xsect_t> get_xsect_edges_queue() {
    auto& stream = ctx_.get_stream();
    return ArrayView<xsect_t>(xsect_edges_.data(), xsect_edges_.size(stream));
  }

 private:
  RTQueryConfig get_rt_query_config(int map_id) const {
    RTQueryConfig pip_config;

    pip_config.use_triangle = config_.use_triangle;
    pip_config.fau = config_.fau;
    pip_config.epsilon = config_.epsilon;
    pip_config.early_term_deviant = config_.early_term_deviant;
    pip_config.handle_ = traverse_handles_[map_id];
    return pip_config;
  }

  CONTEXT_T& ctx_;
  OverlayConfig config_;
  std::shared_ptr<RTEngine> rt_engine_;
  // Algo
  PIPRT<CONTEXT_T> pip_;
  uint32_t max_n_xsects_{};
  Queue<xsect_t> xsect_edges_;
  thrust::device_vector<xsect_t> xsect_edges_sorted_[2];
  // point->polygon id
  thrust::device_vector<polygon_id_t> point_in_polygon_[2];
  OptixTraversableHandle traverse_handles_[2];
  // RT
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<float3> triangle_points_;

  // Only for benchmark

#ifndef NDEBUG
  thrust::device_vector<uint32_t> hit_count_[2];
  thrust::device_vector<uint32_t> closer_count_[2];
  thrust::device_vector<uint32_t> above_edge_count_[2];
  thrust::device_vector<uint32_t> fail_update_count_[2];
#endif
};

}  // namespace rayjoin

#endif  // RAYJOIN_APP_RT_OVERLAY_H
