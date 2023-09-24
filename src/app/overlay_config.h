#ifndef APP_OVERLAY_CONFIG_H
#define APP_OVERLAY_CONFIG_H

#include <string>
namespace rayjoin {

struct OverlayConfig {
  std::string exec_root;
  std::string map1_path;
  std::string map2_path;
  std::string serialize_prefix;
  std::string output_path;
  unsigned int grid_size;
  float xsect_factor;
  std::string mode;
  bool fau;
  bool check;
  bool profile;
  bool lb;
  int win;
  bool new_comp;
  int compress_iter;
  float enlarge;
};
}  // namespace rayjoin

#endif  // APP_OVERLAY_CONFIG_H
