#ifndef APP_MAP_OVERLAY_RT_H
#define APP_MAP_OVERLAY_RT_H
#include <iomanip>
#include <random>

#include "app/lsi_rt.h"
#include "app/map_overlay.h"
#include "app/output_chain.h"
#include "app/overlay_config.h"
#include "app/pip_rt.h"
#include "rt/primitive.h"
#include "rt/rt_engine.h"
#include "util/helpers.h"
#include "util/stopwatch.h"
#include "util/util.h"

namespace rayjoin {

template <typename CONTEXT_T>
class MapOverlayRT : public MapOverlay<CONTEXT_T> {
  using coord_t = typename CONTEXT_T::coord_t;
  using internal_coord_t = typename CONTEXT_T::internal_coord_t;
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using xsect_t = dev::Intersection<internal_coord_t>;

 public:
  explicit MapOverlayRT(CONTEXT_T& ctx) : MapOverlay<CONTEXT_T>(ctx) {
    rt_engine_ = std::make_shared<RTEngine>();
    this->lsi_ = std::make_shared<LSIRT<CONTEXT_T>>(ctx, rt_engine_);
    this->pip_ = std::make_shared<PIPRT<CONTEXT_T>>(ctx, rt_engine_);
  }

  void set_config(const QueryConfigRT& config) { config_ = config; }

  void Init() override {
    auto& ctx = this->ctx_;
    auto xsect_factor = config_.xsect_factor;
    auto exec_root = ctx.get_exec_root();
    RTConfig rt_config = get_default_rt_config(exec_root);
    size_t max_n_points = 0;
    size_t max_n_edges = 0;

    FOR2 {
      auto map = ctx.get_map(im);
      auto np = map->get_points_num();
      auto ne = map->get_edges_num();

      this->closest_eids_[im].resize(np, DONTKNOW);
      this->point_in_polygon_[im].resize(np, DONTKNOW);
      eid_range_[im] = std::make_shared<
          thrust::device_vector<thrust::pair<size_t, size_t>>>();
      eid_range_[im]->reserve(ne);
      max_n_points = std::max(max_n_points, np);
      max_n_edges = std::max(max_n_edges, ne);
    }
    aabbs_.reserve(max_n_edges);
    size_t n_edges = 0;
    FOR2 { n_edges += ctx.get_map(im)->get_edges_num(); }

    rt_engine_->Init(rt_config);
    this->lsi_->Init(n_edges * xsect_factor);
    this->pip_->Init(max_n_points);
  }

  void BuildIndex() override {
    auto& ctx = this->ctx_;
    auto& stream = ctx.get_stream();
    const auto& scaling = ctx.get_scaling();
    auto win_size = config_.win;
    auto ag_iter = config_.ag_iter;
    auto area_enlarge = config_.enlarge;

    FOR2 {
      auto d_map = ctx.get_map(im)->DeviceObject();
      if (config_.ag == 0) {
        FillPrimitives(stream, d_map, scaling, aabbs_);
      } else if (config_.ag == 1) {
        FillPrimitivesGroupNew(stream, d_map, scaling, ag_iter,
                               area_enlarge, aabbs_, *eid_range_[im]);
      } else if (config_.ag == 2) {
        FillPrimitivesGroup(stream, d_map, scaling, win_size, area_enlarge,
                            aabbs_, *eid_range_[im]);
      }
      traverse_handles_[im] =
          rt_engine_->BuildAccelCustom(stream, ArrayView<OptixAabb>(aabbs_));

      stream.Sync();
      if (config_.fau) {
        aabbs_.resize(0);
        aabbs_.shrink_to_fit();
      }
    };
  }

  void IntersectEdge(int query_map_id) override {
    int base_map_id = 1 - query_map_id;
    auto& stream = this->ctx_.get_stream();
    auto lsi = std::dynamic_pointer_cast<LSIRT<CONTEXT_T>>(this->lsi_);

    config_.eid_range = eid_range_[base_map_id];
    config_.handle = traverse_handles_[base_map_id];

    lsi->set_config(config_);
    lsi->Query(stream, query_map_id);
    stream.Sync();
  }

  void LocateVerticesInOtherMap(int query_map_id) override {
    auto& ctx = this->ctx_;
    Stream& stream = ctx.get_stream();
    auto pip = std::dynamic_pointer_cast<PIPRT<CONTEXT_T>>(this->pip_);
    auto base_map_id = 1 - query_map_id;
    auto d_base_map = ctx.get_map(base_map_id)->DeviceObject();
    auto d_query_map = ctx.get_map(query_map_id)->DeviceObject();
    auto d_points = d_query_map.get_points();

    config_.eid_range = eid_range_[base_map_id];
    config_.handle = traverse_handles_[base_map_id];

    pip->set_config(config_);
    pip->Query(stream, query_map_id, d_points);

    auto& closest_eid = this->pip_->get_closest_eids();

    thrust::copy(thrust::cuda::par.on(stream.cuda_stream()),
                 closest_eid.begin(), closest_eid.end(),
                 this->closest_eids_[query_map_id].begin());

    thrust::transform(thrust::cuda::par.on(stream.cuda_stream()),
                      closest_eid.begin(), closest_eid.end(),
                      this->point_in_polygon_[query_map_id].begin(),
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

  void DumpStatistics(const char* path) {
    std::dynamic_pointer_cast<PIPRT<CONTEXT_T>>(this->pip_)
        ->DumpStatistics(path);
  }

  void ComputeOutputPolygons() override {
    using point_t = typename CONTEXT_T::map_t::point_t;
    auto& ctx = this->ctx_;
    auto& stream = ctx.get_stream();
    auto pip = std::dynamic_pointer_cast<PIPRT<CONTEXT_T>>(this->pip_);
    auto xsects = this->lsi_->get_xsects();
    size_t n_xsects = xsects.size();
    const auto& scaling = ctx.get_scaling();

    FOR2 {
      // TODO: Move them out out the loop
      thrust::device_vector<index_t> unique_eids;
      thrust::device_vector<uint32_t> n_xsects_per_edge;
      thrust::device_vector<uint32_t> xsect_index;
      thrust::device_vector<point_t> mid_points;

      auto& xsect_edges_sorted = xsect_edges_sorted_[im];
      // Sort by eid1, eid2 respectively, so we can do binary search
      xsect_edges_sorted.resize(n_xsects);
      thrust::copy(thrust::cuda::par.on(stream.cuda_stream()), xsects.data(),
                   xsects.data() + n_xsects, xsect_edges_sorted.begin());
      thrust::sort(
          thrust::cuda::par.on(stream.cuda_stream()),
          xsect_edges_sorted.begin(), xsect_edges_sorted.end(),
          [=] __device__(const xsect_t& xsect1, const xsect_t& xsect2) {
            return xsect1.eid[im] < xsect2.eid[im];
          });

      ArrayView<xsect_t> d_xsect_edges_sorted(xsect_edges_sorted);
      auto query_map_id = im, base_map_id = 1 - im;
      auto d_query_map = ctx.get_map(query_map_id)->DeviceObject();
      auto d_base_map = ctx.get_map(base_map_id)->DeviceObject();

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

      config_.eid_range = eid_range_[base_map_id];
      config_.handle = traverse_handles_[base_map_id];

      pip->set_config(config_);
      pip->Query(stream, query_map_id, d_mid_points);

      stream.Sync();

      ArrayView<index_t> d_mid_point_closest_eid(pip->get_closest_eids());

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
    WriteOutputChain(this->ctx_, xsect_edges_sorted_, this->point_in_polygon_,
                     path);
  }

 private:
  std::shared_ptr<RTEngine> rt_engine_;
  QueryConfigRT config_;
  // Algo
  thrust::device_vector<xsect_t> xsect_edges_sorted_[2];
  // point->polygon id
  OptixTraversableHandle traverse_handles_[2];
  // RT
  thrust::device_vector<OptixAabb> aabbs_;
  std::shared_ptr<thrust::device_vector<thrust::pair<size_t, size_t>>>
      eid_range_[2];
};

}  // namespace rayjoin

#endif  // APP_MAP_OVERLAY_RT_H
