
#ifndef APP_RT_QUERY_CONFIG_H
#define APP_RT_QUERY_CONFIG_H
namespace rayjoin {

struct RTQueryConfig {
  bool use_triangle = false;
  bool fau = true;
  int rounding_iter = 2;
  bool reorder = false;
  OptixTraversableHandle handle_;
};
}  // namespace rayjoin
#endif  // APP_RT_QUERY_CONFIG_H
