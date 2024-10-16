#include <glog/logging.h>

#include <algorithm>

#include "flags.h"
#include "run_overlay.cuh"

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  gflags::SetUsageMessage("Usage: -poly1 -poly2");
  if (argc == 1) {
    gflags::ShowUsageWithFlags(argv[0]);
    exit(1);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  rayjoin::OverlayConfig config;
  std::string exec_path = argv[0];

  config.map1_path = FLAGS_poly1;
  config.map2_path = FLAGS_poly2;
  config.output_path = FLAGS_output;
  config.serialize_prefix = FLAGS_serialize;
  config.grid_size = FLAGS_grid_size;
  config.xsect_factor = FLAGS_xsect_factor;
  config.mode = FLAGS_mode;
  config.exec_root = exec_path.substr(0, exec_path.find_last_of("/"));
  config.check = FLAGS_check;
  config.fau = FLAGS_fau;
  config.profile = FLAGS_profile;
  config.win = FLAGS_win;
  config.ag = FLAGS_ag;
  config.ag_iter = FLAGS_ag_iter;
  config.enlarge = FLAGS_enlarge;

  rayjoin::RunOverlay(config);

  google::ShutdownGoogleLogging();
}