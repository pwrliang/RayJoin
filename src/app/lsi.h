#ifndef APP_LSI_H
#define APP_LSI_H
#include "map/map.h"
#include "util/queue.h"
#include "util/type_traits.h"
namespace rayjoin {
template <typename CONTEXT_T>
class LSI {
 protected:
  using coord_t = typename CONTEXT_T::coord_t;
  using internal_coord_t = typename CONTEXT_T::internal_coord_t;
  using coefficient_t = typename CONTEXT_T::coefficient_t;

 public:
  using xsect_t = dev::Intersection<internal_coord_t>;

  explicit LSI(CONTEXT_T& ctx) : ctx_(ctx) {}

  virtual ~LSI() = default;

  virtual void Init(size_t max_n_xsects) {
    LOG(INFO) << "Queue size: " << max_n_xsects * sizeof(xsect_t) / 1024 / 1024
              << " MB";
    xsect_queue_.Init(max_n_xsects);
  }

  virtual void Query(Stream& stream, int query_map_id) = 0;

  CONTEXT_T& get_context() { return ctx_; }

  const CONTEXT_T& get_context() const { return ctx_; }

  ArrayView<xsect_t> get_xsects() { return xsects_; }

 protected:
  CONTEXT_T& ctx_;
  Queue<xsect_t> xsect_queue_;
  ArrayView<xsect_t> xsects_;
  SharedValue<uint64_t> prof_counter_;
};

}  // namespace rayjoin

#endif  // APP_LSI_H
