
#ifndef APP_RT_QUERY_CONFIG_H
#define APP_RT_QUERY_CONFIG_H
namespace rayjoin {

struct RTQueryConfig {
  bool fau = true;
  int rounding_iter = 2;
  int win_size = 1;
  float enlarge = 0;
  bool reorder = false;
  thrust::pair<size_t, size_t>* eid_range = nullptr;
  OptixTraversableHandle handle_ = 0;
};
}  // namespace rayjoin
#endif  // APP_RT_QUERY_CONFIG_H
