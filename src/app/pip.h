#ifndef APP_PIP_H
#define APP_PIP_H

#include "map/map.h"
#include "util/queue.h"
#include "util/type_traits.h"
namespace rayjoin {
template <typename CONTEXT_T>
class PIP {
  using coord_t = typename CONTEXT_T::coord_t;
  using internal_coord_t = typename CONTEXT_T::internal_coord_t;
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using map_t = typename CONTEXT_T::map_t;
  using point_t = typename map_t::point_t;

 public:
  explicit PIP(CONTEXT_T& ctx) : ctx_(ctx) {}

  virtual ~PIP() = default;

  virtual void Init(size_t n_points) { polygon_ids_.reserve(n_points); }

  virtual thrust::device_vector<polygon_id_t>& Query(
      int map_id, ArrayView<point_t> points) = 0;

  CONTEXT_T& get_context() { return ctx_; }

  const CONTEXT_T& get_context() const { return ctx_; }

 protected:
  CONTEXT_T& ctx_;
  thrust::device_vector<polygon_id_t> polygon_ids_;
  SharedValue<uint64_t> prof_counter_;
};

}  // namespace rayjoin

#endif  // APP_PIP_H
