#ifndef APP_MAP_OVERLAY_GRID_H
#define APP_MAP_OVERLAY_GRID_H
#include "app/lsi_grid.h"
#include "app/map_overlay.h"
#include "app/output_chain.h"
#include "app/pip_grid.h"
#include "app/query_config.h"
#include "grid/uniform_grid.h"

namespace rayjoin {

template <typename CONTEXT_T>
class MapOverlayGrid : MapOverlay<CONTEXT_T> {
  using coord_t = typename CONTEXT_T::coord_t;
  using internal_coord_t = typename CONTEXT_T::internal_coord_t;
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using xsect_t = dev::Intersection<internal_coord_t>;

 public:
  explicit MapOverlayGrid(CONTEXT_T& ctx) : MapOverlay<CONTEXT_T>(ctx) {}

  void set_query_config(const QueryConfigGrid& config) { config_ = config; }

  void Init() override {
    auto& ctx = this->ctx_;
    auto gsize = config_.grid_size;
    auto xsect_factor = config_.xsect_factor;
    size_t n_edges = 0;
    size_t max_n_points = 0;

    FOR2 {
      auto map = this->ctx_.get_map(im);
      auto points_num = map->get_points_num();
      this->point_in_polygon_[im].resize(points_num, DONTKNOW);
      n_edges += map->get_edges_num();
    }

    grid_ = std::make_shared<UniformGrid>(gsize);
    this->lsi_ = std::make_shared<LSIGrid<CONTEXT_T>>(ctx, grid_);
    this->pip_ = std::make_shared<PIPGrid<CONTEXT_T>>(ctx, grid_);
    this->lsi_->Init(xsect_factor * n_edges);
    this->pip_->Init(max_n_points);  // allocate space
  }

  void BuildIndex() override { grid_->AddMapsToGrid(this->ctx_); }

  void IntersectEdge(int query_map_id) override {
    auto& stream = this->ctx_.get_stream();
    auto lsi = std::dynamic_pointer_cast<LSIGrid<CONTEXT_T>>(this->lsi_);

    lsi->set_config(config_);
    lsi->Query(stream, query_map_id);
    xsect_edges_ = lsi->get_xsects();
    LOG(INFO) << "Xsects: " << xsect_edges_.size();
  }

  void LocateVerticesInOtherMap(int query_map_id) override {
    auto& ctx = this->ctx_;
    auto& stream = ctx.get_stream();
    int base_map_id = 1 - query_map_id;
    auto d_base_map = ctx.get_map(base_map_id)->DeviceObject();
    auto d_query_map = ctx.get_map(query_map_id)->DeviceObject();
    auto d_points = d_query_map.get_points();

    this->pip_->Query(stream, query_map_id, d_points);

    auto& closest_eid = this->pip_->get_closest_eids();

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

  void ComputeOutputPolygons() override {
    auto& ctx = this->ctx_;
    auto& stream = ctx.get_stream();
    size_t n_xsects = xsect_edges_.size();
    const auto& scaling = ctx.get_scaling();
    auto d_grid = grid_->DeviceObject();
    thrust::device_vector<int64_t> unique_eids;

    FOR2 {
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
      auto d_map = ctx.get_map(im)->DeviceObject();
      auto dst_map_id = 1 - im;
      auto d_dst_map = ctx.get_map(dst_map_id)->DeviceObject();

      unique_eids.resize(n_xsects);

      thrust::transform(
          thrust::cuda::par.on(stream.cuda_stream()),
          xsect_edges_sorted.begin(), xsect_edges_sorted.end(),
          unique_eids.begin(),
          [=] __device__(const xsect_t& xsect) { return xsect.eid[im]; });
      auto end = thrust::unique(thrust::cuda::par.on(stream.cuda_stream()),
                                unique_eids.begin(), unique_eids.end());

      unique_eids.resize(end - unique_eids.begin());

      ArrayView<int64_t> d_unique_eids(unique_eids);

      ForEach(stream, d_unique_eids.size(), [=] __device__(size_t idx) mutable {
        auto gsize = d_grid.get_grid_size();
        auto eid = d_unique_eids[idx];
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
        auto n_xsect = it.second - it.first;

        // if current edge intersects more than 1 edges in other map
        if (n_xsect > 1) {
          const auto& e = d_map.get_edge(eid);
          const auto& p1 = d_map.get_point(e.p1_idx);

          // sort intersection points on 2-d plane
          thrust::sort(
              thrust::seq, it.first, it.second,
              [=](const xsect_t& xsect1, const xsect_t& xsect2) {
                // using __int128 in case of overflow
                auto d1 = SQ(xsect1.x - tcb::rational<__int128>(p1.x)) +
                          SQ(xsect1.y - tcb::rational<__int128>(p1.y));
                auto d2 = SQ(xsect2.x - tcb::rational<__int128>(p1.x)) +
                          SQ(xsect2.y - tcb::rational<__int128>(p1.y));
                return d1 < d2;
              });
          // access consecutive intersection points
          for (int xsect_idx = 0; xsect_idx < n_xsect - 1; xsect_idx++) {
            xsect_t& xsect1 = *(it.first + xsect_idx);
            xsect_t& xsect2 = *(it.first + xsect_idx + 1);
            tcb::rational<__int128> x1 = xsect1.x, y1 = xsect1.y;
            tcb::rational<__int128> x2 = xsect2.x, y2 = xsect2.y;
            dev::ExactPoint<internal_coord_t> mid_p(x1 + (x2 - x1) / 2,
                                                    y1 + (y2 - y1) / 2);

            polygon_id_t ipol = EXTERIOR_FACE_ID;
            auto cx = dev::calculate_cell(gsize, scaling, mid_p.x);
            auto cy = dev::calculate_cell(gsize, scaling, mid_p.y);

            assert(cx >= 0 && cx < gsize);
            assert(cy >= 0 && cy < gsize);

            // get face id of mid point
            // by shooting a ray toward positive y-axis
            for (auto curr_cy = cy; curr_cy < gsize; curr_cy++) {
              // find a closest edge of dst-map to this vertex in this cell
              auto* beste = dev::test_against_edges_of_this_cell(
                  mid_p,    /* IN: current vertex */
                  cx,       /* IN: Current cell x */
                  curr_cy,  /* IN: Current cell y */
                  d_grid,   /* IN: Current Cell */
                  scaling,  /* IN: Scaling */
                  d_dst_map /* IN: Map to test */
              );
              if (beste != nullptr) {
                ipol = d_dst_map.get_face_id(*beste);
                break;
              }
            }
            xsect1.mid_point_polygon_id = ipol;
          }
        }
      });
    }
    stream.Sync();
  }

  void WriteResult(const char* path) override {
    WriteOutputChain(this->ctx_, xsect_edges_sorted_, point_in_polygon_, path);
  }

  const thrust::device_vector<xsect_t>& get_xsect_edges(int im) const {
    return xsect_edges_sorted_[im];
  }

  const thrust::device_vector<index_t>& get_closet_eids() const {
    return this->pip_->get_closest_eids();
  }

  ArrayView<xsect_t> get_xsect_edges() const { return xsect_edges_; }

 private:
  std::shared_ptr<UniformGrid> grid_;
  QueryConfigGrid config_;
  ArrayView<xsect_t> xsect_edges_;
  thrust::device_vector<xsect_t> xsect_edges_sorted_[2];
  thrust::device_vector<polygon_id_t> point_in_polygon_[2];
};

}  // namespace rayjoin

#endif  // APP_MAP_OVERLAY_GRID_H
