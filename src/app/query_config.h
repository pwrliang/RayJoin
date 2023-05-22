#ifndef APP_RT_QUERY_CONFIG_H
#define APP_RT_QUERY_CONFIG_H
#include "tree/primtive.h"

namespace rayjoin {
struct QueryConfigRT {
  bool profiling = false;
  bool fau = true;
  int rounding_iter = 2;
  OptixTraversableHandle handle;
  std::shared_ptr<thrust::device_vector<thrust::pair<size_t, size_t>>>
      eid_range;
};

struct QueryConfigLBVH {
  bool profiling = false;
  std::shared_ptr<lbvh::bvh<float, segment, aabb_getter>> lbvh;
};
}  // namespace rayjoin
#endif  // APP_RT_QUERY_CONFIG_H
