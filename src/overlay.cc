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
  config.epsilon = FLAGS_epsilon;
  config.rounding_iter = FLAGS_rounding_iter;
  config.mode = FLAGS_mode;
  config.exec_root = exec_path.substr(0, exec_path.find_last_of("/"));
  config.check = FLAGS_check;
  config.lb = FLAGS_lb;
  config.use_triangle = FLAGS_triangle;
  config.early_term_deviant = FLAGS_early_term_deviant;
  config.fau = FLAGS_fau;
  config.profiling = FLAGS_profile;

  rayjoin::RunOverlay(config);

  google::ShutdownGoogleLogging();
}