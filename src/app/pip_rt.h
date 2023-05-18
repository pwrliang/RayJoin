
#ifndef APP_PIP_RT_H
#define APP_PIP_RT_H
#include <thrust/host_vector.h>

#include <fstream>
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
  explicit PIPRT(CONTEXT_T& ctx, std::shared_ptr<RTEngine> rt_engine)
      : PIP<CONTEXT_T>(ctx), rt_engine_(std::move(rt_engine)) {}

  virtual ~PIPRT() = default;

  void set_query_config(const RTQueryConfig& query_config) {
    query_config_ = query_config;
  }

  void Query(Stream& stream, int base_map_id,
             ArrayView<point_t> d_query_points) {
    auto& ctx = this->ctx_;
    auto& scaling = ctx.get_scaling();
    auto d_base_map = ctx.get_map(base_map_id)->DeviceObject();
    auto points_num = d_query_points.size();
    auto module_id = ModuleIdentifier::MODULE_ID_PIP_CUSTOM;

    this->closest_eids_.resize(points_num);

    thrust::fill(this->closest_eids_.begin(), this->closest_eids_.end(),
                 DONTKNOW);

    LaunchParamsPIP params;
    params.base_map_edges = d_base_map.get_edges().data();
    params.base_map_points = d_base_map.get_points().data();
    params.eid_range = query_config_.eid_range;
    params.query_map_id = 1 - base_map_id;
    params.query_points = d_query_points;
    params.scaling = scaling;
    params.traversable = query_config_.handle_;
    params.closest_eids = thrust::raw_pointer_cast(this->closest_eids_.data());
#ifndef NDEBUG
    hit_count_.resize(points_num, 0);
    closer_count_.resize(points_num, 0);
    last_update_count_.resize(points_num, 0);
    fail_update_count_.resize(points_num, 0);

    params.hit_count = ArrayView<uint32_t>(hit_count_).data();
    params.closer_count = ArrayView<uint32_t>(closer_count_).data();
    params.last_update_count = ArrayView<uint32_t>(last_update_count_).data();
    params.fail_update_count = ArrayView<uint32_t>(fail_update_count_).data();
#endif
    rt_engine_->CopyLaunchParams(stream, params);
    rt_engine_->Render(stream, module_id,
                       dim3{(unsigned int) points_num, 1, 1});
    stream.Sync();
  }

  void DumpStatistics(const char* path) {
#ifndef NDEBUG
    thrust::host_vector<uint32_t> hit_count = hit_count_;
    thrust::host_vector<uint32_t> closer_count = closer_count_;
    thrust::host_vector<uint32_t> last_update_count = last_update_count_;
    thrust::host_vector<uint32_t> fail_update_count = fail_update_count_;

    std::ofstream ofs(path);

    ofs << "point_idx,hit_count,closer_count,last_update_count,fail_update_"
           "count\n";

    for (size_t point_idx = 0; point_idx < hit_count.size(); point_idx++) {
      ofs << point_idx << "," << hit_count[point_idx] << ","
          << closer_count[point_idx] << "," << last_update_count[point_idx]
          << "," << fail_update_count[point_idx] << "\n";
    }

    ofs.close();
#else
    LOG(FATAL) << "DumpStatistics for PIP only works in debug mode";
#endif
  }

  std::shared_ptr<RTEngine> get_rt_engine() { return rt_engine_; }

 protected:
  std::shared_ptr<RTEngine> rt_engine_;
  RTQueryConfig query_config_;
#ifndef NDEBUG
  thrust::device_vector<uint32_t> hit_count_;
  thrust::device_vector<uint32_t> closer_count_;
  thrust::device_vector<uint32_t> last_update_count_;
  thrust::device_vector<uint32_t> fail_update_count_;
#endif
};

}  // namespace rayjoin

#endif  // APP_PIP_RT_H
