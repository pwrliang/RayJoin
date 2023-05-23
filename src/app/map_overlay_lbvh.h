#ifndef APP_MAP_OVERLAY_LBVH_H
#define APP_MAP_OVERLAY_LBVH_H
#include "app/lsi_lbvh.h"
#include "app/output_chain.h"
#include "app/pip_lbvh.h"

namespace rayjoin {

template <typename CONTEXT_T>
class MapOverlayLBVH {
  using coord_t = typename CONTEXT_T::coord_t;
  using internal_coord_t = typename CONTEXT_T::internal_coord_t;
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using xsect_t = dev::Intersection<internal_coord_t>;

 public:
  explicit MapOverlayLBVH(CONTEXT_T& ctx, float xsect_factor)
      : ctx_(ctx), lsi_(ctx), pip_(ctx), xsect_factor_(xsect_factor) {}

  void Init() {
    size_t n_edges = 0;
    size_t max_n_points = 0;

    FOR2 {
      auto map = ctx_.get_map(im);
      auto points_num = map->get_points_num();
      point_in_polygon_[im].resize(points_num, DONTKNOW);
      n_edges += map->get_edges_num();
    }
    lsi_.Init(xsect_factor_ * n_edges);
    pip_.Init(max_n_points);  // allocate space
  }

  void BuildBVH(int map_id) {
    auto& stream = ctx_.get_stream();
    const auto& scaling = ctx_.get_scaling();
    auto d_map = ctx_.get_map(map_id)->DeviceObject();
    thrust::device_vector<segment> primitives;

    auto bvh = std::make_shared<lbvh::bvh<float, segment, aabb_getter>>();
    FillPrimitivesLBVH(stream, d_map, scaling, primitives);
    stream.Sync();
    bvh->assign(primitives);
    bvh->construct();
    lbvhs_[map_id] = bvh;
  }

  void IntersectEdge() {
    auto& stream = ctx_.get_stream();
    int base_map_id = 0, query_map_id = 1 - base_map_id;

    QueryConfigLBVH query_config = get_query_config(base_map_id);
    lsi_.set_config(query_config);
    lsi_.Query(stream, query_map_id);
    xsect_edges_ = lsi_.get_xsects();
    LOG(INFO) << "Xsects: " << xsect_edges_.size();
  }

  void LocateVerticesInOtherMap(int query_map_id) {
    auto& stream = ctx_.get_stream();
    int base_map_id = 1 - query_map_id;
    auto d_base_map = ctx_.get_map(base_map_id)->DeviceObject();
    auto d_query_map = ctx_.get_map(query_map_id)->DeviceObject();
    auto d_points = d_query_map.get_points();
    auto config = get_query_config(base_map_id);

    pip_.set_config(config);
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

  void ComputeOutputPolygons() {
    using point_t = typename CONTEXT_T::map_t::point_t;
    auto& stream = ctx_.get_stream();
    size_t n_xsects = xsect_edges_.size();
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

      auto query_config = get_query_config(base_map_id);

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

  const thrust::device_vector<xsect_t>& get_xsect_edges(int im) const {
    return xsect_edges_sorted_[im];
  }

  const thrust::device_vector<index_t>& get_closet_eids() const {
    return pip_.get_closest_eids();
  }

  ArrayView<xsect_t> get_xsect_edges() const { return xsect_edges_; }

 private:
  QueryConfigLBVH get_query_config(int base_map_id) {
    QueryConfigLBVH config;
    config.lbvh = lbvhs_[base_map_id];
    return config;
  }

  CONTEXT_T& ctx_;
  LSILBVH<CONTEXT_T> lsi_;
  LSILBVH<CONTEXT_T> pip_;
  float xsect_factor_;
  ArrayView<xsect_t> xsect_edges_;
  thrust::device_vector<xsect_t> xsect_edges_sorted_[2];
  thrust::device_vector<polygon_id_t> point_in_polygon_[2];
  std::shared_ptr<lbvh::bvh<float, segment, aabb_getter>> lbvhs_[2];
};

}  // namespace rayjoin

#endif  // APP_MAP_OVERLAY_GRID_H
