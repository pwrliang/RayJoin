#ifndef APP_RT_QUERY_CONFIG_H
#define APP_RT_QUERY_CONFIG_H
#include "tree/primtive.h"

namespace rayjoin {
struct QueryConfigGrid {
  unsigned int grid_size = 2048;
  bool profiling = false;
  float xsect_factor = 1.5;
  bool lb = true;
};

struct QueryConfigRT {
  bool profiling = false;
  bool fau = true;
  float xsect_factor = 1.5;
  int win;
  float enlarge;
  int rounding_iter = 2;
  OptixTraversableHandle handle;
  std::shared_ptr<thrust::device_vector<thrust::pair<size_t, size_t>>>
      eid_range;
};

struct QueryConfigLBVH {
  bool profiling = false;
  float xsect_factor = 1.5;
  std::shared_ptr<lbvh::bvh<float, segment, aabb_getter>> lbvh;
};

}  // namespace rayjoin
#endif  // APP_RT_QUERY_CONFIG_H
