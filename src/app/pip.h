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

  virtual void Init(size_t n_points) { closest_eids_.reserve(n_points); }

  virtual void Query(Stream& stream, int base_map_id,
                     ArrayView<point_t> query_points) = 0;

  CONTEXT_T& get_context() { return ctx_; }

  const CONTEXT_T& get_context() const { return ctx_; }

  const thrust::device_vector<index_t>& get_closest_eids() const {
    return closest_eids_;
  }

 protected:
  CONTEXT_T& ctx_;
  thrust::device_vector<index_t> closest_eids_;
  SharedValue<uint64_t> prof_counter_;
};

}  // namespace rayjoin

#endif  // APP_PIP_H
