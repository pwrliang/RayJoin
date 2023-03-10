#ifndef RUN_QUERY_CUH
#define RUN_QUERY_CUH
namespace rayjoin {
struct QueryConfig {
  std::string exec_root;
  std::string map_path;
  std::string output_path;
  std::string serialize_prefix;
  int warmup;
  int repeat;
  unsigned int grid_size;
  bool dynamic_query;
  std::string sample;
  float sample_rate;
  float xsect_factor;
  float gen_t;  // edge len
  int gen_n;
  int random_seed;
  std::string mode;
  bool check;
  bool lb;
  bool use_triangle;
  bool print_histo;
  bool profiling;
  double epsilon;
  bool fau;
  float early_term_deviant;
};

void RunLSIQuery(const QueryConfig& config);
void RunPIPQuery(const QueryConfig& config);
}  // namespace rayjoin
#endif  // RUN_QUERY_CUH
