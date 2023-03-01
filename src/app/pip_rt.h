
#ifndef APP_PIP_RT_H
#define APP_PIP_RT_H
#include <utility>

#include "app/rt_query_config.h"
#include "map/map.h"
#include "rt/rt_engine.h"
#include "util/queue.h"
#include "util/type_traits.h"

namespace rayjoin {

template <typename CONTEXT_T>
class PIPRT : public PIP<CONTEXT_T> {
  using coord_t = typename CONTEXT_T::coord_t;
  using internal_coord_t = typename CONTEXT_T::internal_coord_t;
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using map_t = typename CONTEXT_T::map_t;
  using point_t = typename map_t::point_t;

 public:
  explicit PIPRT(CONTEXT_T& ctx, std::shared_ptr<RTEngine> rt_engine,
                 RTQueryConfig config)
      : PIP<CONTEXT_T>(ctx),
        rt_engine_(std::move(rt_engine)),
        config_(config) {}

  virtual ~PIPRT() = default;

  void BuildIndex(int map_id) {
    auto& ctx = this->ctx_;
    auto& stream = ctx.get_stream();
    const auto& scaling = ctx.get_scaling();
    auto d_map = ctx.get_map(map_id)->DeviceObject();
    auto ne = d_map.get_edges_num();
    auto epsilon = config_.epsilon;

    if (config_.use_triangle) {
      auto bb = ctx.get_bounding_box();
      auto mid_x = (bb.min_x + bb.max_x) / 2.0;
      auto mid_y = (bb.min_y + bb.max_y) / 2.0;
      auto delta_x = scaling.ScaleX(mid_x + epsilon) - scaling.ScaleX(mid_x);
      auto delta_y = scaling.ScaleY(mid_y + epsilon) - scaling.ScaleY(mid_y);

      triangle_points_.resize(d_map.get_edges_num() * 3);

      ArrayView<float3> d_triangle_points(triangle_points_);

      ForEach(stream, ne, [=] __device__(uint32_t eid) mutable {
        const auto& e = d_map.get_edge(eid);
        auto p1_idx = e.p1_idx;
        auto p2_idx = e.p2_idx;
        auto p1 = d_map.get_point(p1_idx);
        auto p2 = d_map.get_point(p2_idx);
        auto x1 = scaling.UnscaleX(p1.x);
        auto y1 = scaling.UnscaleY(p1.y);
        auto x2 = scaling.UnscaleX(p2.x);
        auto y2 = scaling.UnscaleY(p2.y);

        if (e.b == 0) {
          assert(p1.x == p2.x);
          if (p1.y > p2.y) {
            SWAP(p1, p2);
          }

          x1 = x2 = scaling.UnscaleX(p1.x);
          y1 = scaling.UnscaleY(p1.y - delta_y);
          y2 = scaling.UnscaleY(p2.y + delta_y);
        } else {
          assert(p1.x != p2.x);
          if (p1.x > p2.x) {
            SWAP(p1, p2);
          }

          // use double is much faster than rational
          // this does not need to be accurate
          double a = -e.a / e.b;
          double b = -e.c / e.b;

          auto new_x1 = p1.x - delta_x;
          auto new_y1 = a * new_x1 + b;

          auto new_x2 = p2.x + delta_x;
          auto new_y2 = a * new_x2 + b;

          x1 = scaling.UnscaleX(new_x1);
          y1 = scaling.UnscaleY(new_y1);
          x2 = scaling.UnscaleX(new_x2);
          y2 = scaling.UnscaleY(new_y2);
        }

        d_triangle_points[eid * 3] = {(float) x1, (float) y1, 0};
        d_triangle_points[eid * 3 + 1] = {(float) x2, (float) y2, 0};
        d_triangle_points[eid * 3 + 2] = {
            (float) (x1 + x2) / 2, (float) (y1 + y2) / 2, PRIMITIVE_HEIGHT};
      });
      handles_[map_id] =
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

        aabb.minX = min(x1, x2) - epsilon;
        aabb.maxX = max(x1, x2) + epsilon;
        aabb.minY = min(y1, y2) - epsilon;
        aabb.maxY = max(y1, y2) + epsilon;
        aabb.minZ = -PRIMITIVE_HEIGHT / 2;
        aabb.maxZ = PRIMITIVE_HEIGHT / 2;
      });
      handles_[map_id] = rt_engine_->BuildAccelCustom(stream, d_aabbs);
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

  thrust::device_vector<polygon_id_t>& Query(int map_id,
                                             ArrayView<point_t> points) {
    auto& ctx = this->ctx_;
    auto& stream = ctx.get_stream();
    auto& scaling = ctx.get_scaling();
    auto d_map = ctx.get_map(map_id)->DeviceObject();
    auto points_num = points.size();
    auto module_id = config_.use_triangle
                         ? ModuleIdentifier::MODULE_ID_PIP
                         : ModuleIdentifier::MODULE_ID_PIP_CUSTOM;

    this->polygon_ids_.resize(points_num);
    thrust::fill(thrust::cuda::par.on(stream.cuda_stream()),
                 this->polygon_ids_.begin(), this->polygon_ids_.end(),
                 DONTKNOW);

    LaunchParamsPIP params;

    params.im = map_id;
    params.src_points = points;
    params.dst_edges = d_map.get_edges().data();
    params.dst_points = d_map.get_points().data();
    params.scaling = scaling;
    params.traversable = handles_.at(map_id);
    params.early_term_deviant = config_.early_term_deviant;
    params.point_in_polygon =
        thrust::raw_pointer_cast(this->polygon_ids_.data());
#ifndef NDEBUG
    hit_count_.resize(points_num, 0);
    closer_count_.resize(points_num, 0);
    above_edge_count_.resize(points_num, 0);
    fail_update_count_.resize(points_num, 0);

    params.hit_count = ArrayView<uint32_t>(hit_count_).data();
    params.closer_count = ArrayView<uint32_t>(closer_count_).data();
    params.above_edge_count = ArrayView<uint32_t>(above_edge_count_).data();
    params.fail_update_count = ArrayView<uint32_t>(fail_update_count_).data();
#endif

    rt_engine_->CopyLaunchParams(stream, params);
    rt_engine_->Render(stream, module_id,
                       dim3{(unsigned int) points.size(), 1, 1});
    // For custom impl, keeping closest k records
    stream.Sync();
    return this->polygon_ids_;
  }

  std::shared_ptr<RTEngine> get_rt_engine() { return rt_engine_; }

 protected:
  thrust::device_vector<polygon_id_t> polygon_ids_;
  std::shared_ptr<RTEngine> rt_engine_;
  RTQueryConfig config_;
  std::map<int, OptixTraversableHandle> handles_;
  thrust::device_vector<OptixAabb> aabbs_;
  thrust::device_vector<float3> triangle_points_;

#ifndef NDEBUG
  thrust::device_vector<uint32_t> hit_count_;
  thrust::device_vector<uint32_t> closer_count_;
  thrust::device_vector<uint32_t> above_edge_count_;
  thrust::device_vector<uint32_t> fail_update_count_;
#endif
};

}  // namespace rayjoin

#endif  // APP_PIP_RT_H
