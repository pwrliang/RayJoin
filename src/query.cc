#include <glog/logging.h>

#include <algorithm>

#include "flags.h"
#include "run_query.cuh"

int main(int argc, char* argv[]) {
  FLAGS_stderrthreshold = 0;

  gflags::SetUsageMessage("Usage: -poly1");
  if (argc == 1) {
    gflags::ShowUsageWithFlags(argv[0]);
    exit(1);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();

  rayjoin::QueryConfig config;
  std::string exec_path = argv[0];

  config.map_path = FLAGS_poly1;
  config.query_path = FLAGS_poly2;
  config.output_path = FLAGS_output;
  config.serialize_prefix = FLAGS_serialize;
  config.warmup = FLAGS_warmup;
  config.repeat = FLAGS_repeat;
  config.grid_size = FLAGS_grid_size;
  config.xsect_factor = FLAGS_xsect_factor;
  config.mode = FLAGS_mode;
  config.exec_root = exec_path.substr(0, exec_path.find_last_of('/'));
  config.check = FLAGS_check;
  config.lb = FLAGS_lb;
  config.sample_map_id = FLAGS_sample_map_id;
  config.sample = FLAGS_sample;
  config.sample_rate = FLAGS_sample_rate;
  config.random_seed = FLAGS_seed;
  config.gen_t = FLAGS_gen_t;
  config.gen_n = FLAGS_gen_n;
  config.print_histo = FLAGS_histo;
  config.profile = FLAGS_profile;
  config.fau = FLAGS_fau;
  config.ag = FLAGS_ag;
  config.ag_iter = FLAGS_ag_iter;
  config.win = FLAGS_win;
  config.enlarge = FLAGS_enlarge;

  if (FLAGS_query == "lsi") {
    rayjoin::RunLSIQuery(config);
  } else if (FLAGS_query == "pip") {
    rayjoin::RunPIPQuery(config);
  } else {
    LOG(FATAL) << "Invalid query: " << FLAGS_query;
  }

  google::ShutdownGoogleLogging();
}