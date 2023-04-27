#ifndef RAYJOIN_MAP_PLANAR_GRAPH_H
#define RAYJOIN_MAP_PLANAR_GRAPH_H
#include <dirent.h>
#include <glog/logging.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "map/bounding_box.h"
#include "util/type_traits.h"
#include "util/util.h"

#define STREAM_WRITE_VAR(stream, var) \
  stream.write(reinterpret_cast<char*>(&(var)), sizeof(var));
#define STREAM_READ_VAR(stream, var) \
  stream.read(reinterpret_cast<char*>(&var), sizeof(var));
namespace rayjoin {

struct Chain {
  int64_t id;               // chain index
  int64_t first_point_idx;  // unused, first, last index of the chain
  int64_t last_point_idx;
  int64_t left_polygon_id;   // left polygon id of the chain
  int64_t right_polygon_id;  // right polygon id of the chain
};

template <class COORD_T>
struct PlanarGraph {
  using point_t = typename cuda_vec<COORD_T>::type_2d;
  std::vector<Chain> chains;
  pinned_vector<index_t> row_index;  // organized in chains
  pinned_vector<point_t> points;
  BoundingBox<COORD_T> bb;
};

template <typename COORD_T>
inline std::shared_ptr<PlanarGraph<COORD_T>> read_pgraph(const char* path) {
  std::ifstream ifs(path);

  CHECK(ifs.is_open()) << "Cannot open file " << path;

  std::string line;
  Chain* curr_chain;
  int64_t np = 0;
  auto pgraph = std::make_shared<PlanarGraph<COORD_T>>();
  auto& g = *pgraph;
  typename cuda_vec<COORD_T>::type_2d* last_p = nullptr;
  std::vector<double> seg_lens;

  while (std::getline(ifs, line)) {
    if (line.empty() || line[0] == '#' || line[0] == '%') {
      continue;
    }
    std::istringstream iss(line);
    bool bad_line;

    if (np == 0) {
      g.chains.template emplace_back();
      curr_chain = &g.chains.back();

      bad_line = !(iss >> curr_chain->id >> np >> curr_chain->first_point_idx >>
                   curr_chain->last_point_idx >> curr_chain->left_polygon_id >>
                   curr_chain->right_polygon_id);
      bad_line |= np < 2;
      //      bad_line |= curr_chain->left_polygon_id ==
      //      curr_chain->right_polygon_id;
      pgraph->row_index.push_back(g.points.size());
      last_p = nullptr;
    } else {
      typename cuda_vec<COORD_T>::type_2d p;

      bad_line = !(iss >> p.x >> p.y);
      if (last_p != nullptr) {
        auto seg_len = sqrt((p.x - last_p->x) * (p.x - last_p->x) +
                            (p.y - last_p->y) * (p.y - last_p->y));
        seg_lens.push_back(seg_len);
        bad_line |= p.x == last_p->x && p.y == last_p->y;
      }

      g.bb.min_x = std::min(g.bb.min_x, p.x);
      g.bb.max_x = std::max(g.bb.max_x, p.x);
      g.bb.min_y = std::min(g.bb.min_y, p.y);
      g.bb.max_y = std::max(g.bb.max_y, p.y);
      g.points.push_back(p);
      last_p = &g.points.back();
      np--;
    }

    CHECK(!bad_line) << "Cannot parse line " << line;
  }
  ifs.close();

  if (!g.points.empty()) {  // in case of an empty graph
    pgraph->row_index.push_back(g.points.size());
  }
  CHECK_EQ(np, 0);

  double total_seg_len = std::accumulate(seg_lens.begin(), seg_lens.end(), 0.0);
  double mean = total_seg_len / seg_lens.size();

  std::vector<double> diff(seg_lens.size());
  std::transform(seg_lens.begin(), seg_lens.end(), diff.begin(),
                 [mean](double x) { return x - mean; });
  double sq_sum =
      std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / seg_lens.size());

  VLOG(1) << "Map " << path << " is loaded, chains: " << g.chains.size()
          << " points: " << pgraph->points.size()
          << " edges: " << g.points.size() - g.chains.size()
          << ", min seg len: "
          << *std::min_element(seg_lens.begin(), seg_lens.end())
          << ", max seg len: "
          << *std::max_element(seg_lens.begin(), seg_lens.end())
          << ", avg seg len: " << mean << ", stdev: " << stdev;

  return pgraph;
}

template <typename COORD_T>
inline void serialize_pgraph(std::shared_ptr<PlanarGraph<COORD_T>> pgraph,
                             const char* path) {
  std::ofstream ofs;
  ofs.open(path, std::ios::out | std::ios::binary);

  uint64_t n_chains = pgraph->chains.size();
  uint64_t n_row_index = pgraph->row_index.size();
  uint64_t n_points = pgraph->points.size();
  uint64_t check_sum = 0xabcdabcd;

  assert(ofs.good());

  STREAM_WRITE_VAR(ofs, check_sum);
  STREAM_WRITE_VAR(ofs, n_chains);
  STREAM_WRITE_VAR(ofs, n_row_index);
  STREAM_WRITE_VAR(ofs, n_points);

  for (auto& chain : pgraph->chains) {
    STREAM_WRITE_VAR(ofs, chain.id);
    STREAM_WRITE_VAR(ofs, chain.first_point_idx);
    STREAM_WRITE_VAR(ofs, chain.last_point_idx);
    STREAM_WRITE_VAR(ofs, chain.left_polygon_id);
    STREAM_WRITE_VAR(ofs, chain.right_polygon_id);
  }
  for (auto& idx : pgraph->row_index) {
    STREAM_WRITE_VAR(ofs, idx);
  }
  for (auto& p : pgraph->points) {
    STREAM_WRITE_VAR(ofs, p.x);
    STREAM_WRITE_VAR(ofs, p.y);
  }
  STREAM_WRITE_VAR(ofs, pgraph->bb.min_x);
  STREAM_WRITE_VAR(ofs, pgraph->bb.min_y);
  STREAM_WRITE_VAR(ofs, pgraph->bb.max_x);
  STREAM_WRITE_VAR(ofs, pgraph->bb.max_y);
  STREAM_WRITE_VAR(ofs, check_sum);

  ofs.close();
}

template <typename COORD_T>
inline std::shared_ptr<PlanarGraph<COORD_T>> deserialize_pgraph(
    const char* path) {
  std::ifstream ifs;
  ifs.open(path, std::ios::in | std::ios::binary);
  auto pgraph = std::make_shared<PlanarGraph<COORD_T>>();

  uint64_t check_sum;
  uint64_t n_chains;
  uint64_t n_row_index;
  uint64_t n_points;

  assert(ifs.good());

  STREAM_READ_VAR(ifs, check_sum);
  CHECK_EQ(check_sum, 0xabcdabcd);
  STREAM_READ_VAR(ifs, n_chains);
  STREAM_READ_VAR(ifs, n_row_index);
  STREAM_READ_VAR(ifs, n_points);
  pgraph->chains.resize(n_chains);
  pgraph->row_index.resize(n_row_index);
  pgraph->points.resize(n_points);

  for (auto& chain : pgraph->chains) {
    STREAM_READ_VAR(ifs, chain.id);
    STREAM_READ_VAR(ifs, chain.first_point_idx);
    STREAM_READ_VAR(ifs, chain.last_point_idx);
    STREAM_READ_VAR(ifs, chain.left_polygon_id);
    STREAM_READ_VAR(ifs, chain.right_polygon_id);
  }
  for (auto& idx : pgraph->row_index) {
    STREAM_READ_VAR(ifs, idx);
  }
  for (auto& p : pgraph->points) {
    STREAM_READ_VAR(ifs, p.x);
    STREAM_READ_VAR(ifs, p.y);
  }
  STREAM_READ_VAR(ifs, pgraph->bb.min_x);
  STREAM_READ_VAR(ifs, pgraph->bb.min_y);
  STREAM_READ_VAR(ifs, pgraph->bb.max_x);
  STREAM_READ_VAR(ifs, pgraph->bb.max_y);
  STREAM_READ_VAR(ifs, check_sum);
  CHECK_EQ(check_sum, 0xabcdabcd);

  ifs.close();

  VLOG(1) << "Map " << path
          << " is deserialized, chains: " << pgraph->chains.size()
          << " points: " << pgraph->points.size()
          << " edges: " << pgraph->points.size() - pgraph->chains.size();

  return pgraph;
}

template <typename COORD_T>
std::shared_ptr<PlanarGraph<COORD_T>> load_from(
    const std::string& path, const std::string& serialize_prefix) {
  std::string escaped_path;
  std::replace_copy(path.begin(), path.end(), std::back_inserter(escaped_path),
                    '/', '-');
  if (!serialize_prefix.empty()) {
    DIR* dir = opendir(serialize_prefix.c_str());
    if (dir) {
      closedir(dir);
    } else if (ENOENT == errno) {
      if (mkdir(serialize_prefix.c_str(), 0755)) {
        LOG(FATAL) << "Cannot create dir " << path;
      }
    } else {
      LOG(FATAL) << "Cannot open dir " << path;
    }
  }

  auto ser_path = serialize_prefix + '/' + escaped_path + ".bin";

  if (access(ser_path.c_str(), R_OK) == 0) {
    return deserialize_pgraph<COORD_T>(ser_path.c_str());
  }
  auto pgraph = read_pgraph<COORD_T>(path.c_str());
  if (!serialize_prefix.empty() &&
      access(serialize_prefix.c_str(), W_OK) == 0) {
    serialize_pgraph(pgraph, ser_path.c_str());
  }
  return pgraph;
}

template <typename COORD_T>
inline std::shared_ptr<PlanarGraph<COORD_T>> sample_map_from(
    std::shared_ptr<PlanarGraph<COORD_T>> p_graph, float sample_rate,
    int seed = 0) {
  std::random_device rd;
  std::mt19937 gen(seed == 0 ? rd() : seed);

  auto sampled_graph = std::make_shared<PlanarGraph<COORD_T>>();
  auto& bb = sampled_graph->bb;
  std::vector<size_t> pids;

  sampled_graph->chains = p_graph->chains;  // copy chains

  for (size_t i_chain = 0; i_chain < p_graph->chains.size(); i_chain++) {
    auto begin_pid = p_graph->row_index[i_chain];
    auto end_pid = p_graph->row_index[i_chain + 1];

    pids.clear();

    pids.push_back(begin_pid);  // keep first point
    // only sample if the chain has mid-points
    if (end_pid - begin_pid > 2) {
      for (auto pid = begin_pid + 1; pid < end_pid - 1; pid++) {
        pids.push_back(pid);
      }
      std::shuffle(pids.begin() + 1, pids.end(), gen);
      CHECK_GE(pids.size(), 2);

      // only take first "sample rate" pids
      pids.resize(std::max(2ul, (size_t) (pids.size() * sample_rate)));
      // sort pids so the order of coordinates is kept
      std::sort(pids.begin() + 1, pids.end(),
                [&](size_t p1_idx, size_t p2_idx) { return p1_idx < p2_idx; });
    }
    pids.push_back(end_pid - 1);

    sampled_graph->row_index.push_back(sampled_graph->points.size());

    for (auto pid : pids) {
      const auto& p = p_graph->points[pid];

      sampled_graph->points.push_back(p);

      bb.min_x = std::min(bb.min_x, p.x);
      bb.max_x = std::max(bb.max_x, p.x);
      bb.min_y = std::min(bb.min_y, p.y);
      bb.max_y = std::max(bb.max_y, p.y);
    }
  }

  if (!sampled_graph->points.empty()) {
    sampled_graph->row_index.push_back(sampled_graph->points.size());
  }

  VLOG(1) << "Map is sampled, chains: " << sampled_graph->chains.size()
          << " points: " << sampled_graph->points.size() << " edges: "
          << sampled_graph->points.size() - sampled_graph->chains.size();
  return sampled_graph;
}

template <typename COORD_T>
inline std::shared_ptr<PlanarGraph<COORD_T>> sample_edges_from(
    std::shared_ptr<PlanarGraph<COORD_T>> p_graph, float sample_rate,
    int seed = 0) {
  std::vector<std::pair<size_t, size_t>> sampled_edges;

  sampled_edges.reserve(p_graph->points.size());

  for (size_t i_chain = 0; i_chain < p_graph->chains.size(); i_chain++) {
    for (auto pid = p_graph->row_index[i_chain];
         pid < p_graph->row_index[i_chain + 1] - 1; pid++) {
      auto eid = pid - i_chain;

      sampled_edges.emplace_back(i_chain, eid);
    }
  }

  std::random_device rd;
  std::mt19937 gen(seed == 0 ? rd() : seed);

  std::shuffle(sampled_edges.begin(), sampled_edges.end(), gen);

  sampled_edges.resize(sampled_edges.size() * sample_rate);

  std::unordered_map<size_t, std::vector<size_t>> chain_edges;

  // collect eids by chain ids
  for (const auto& e : sampled_edges) {
    auto i_chain = e.first;
    auto eid = e.second;

    chain_edges[i_chain].push_back(eid);
  }

  auto sampled_graph = std::make_shared<PlanarGraph<COORD_T>>();
  auto& bb = sampled_graph->bb;

  sampled_graph->chains.reserve(chain_edges.size());
  size_t n_edges = 0;

  for (auto& e : chain_edges) {
    auto i_chain = e.first;
    const auto& eids = chain_edges.at(i_chain);

    sampled_graph->chains.push_back(p_graph->chains[i_chain]);

    n_edges += eids.size();

    std::vector<size_t> pids;

    for (auto eid : eids) {
      auto p1_idx = eid + i_chain;
      auto p2_idx = p1_idx + 1;

      pids.push_back(p1_idx);
      pids.push_back(p2_idx);
    }

    // sort pids so the order of coordinates is kept
    std::sort(pids.begin(), pids.end(),
              [&](size_t p1_idx, size_t p2_idx) { return p1_idx < p2_idx; });
    pids.resize(std::unique(pids.begin(), pids.end()) - pids.begin());

    sampled_graph->row_index.push_back(sampled_graph->points.size());

    for (auto pid : pids) {
      const auto& p = p_graph->points[pid];

      sampled_graph->points.push_back(p);

      bb.min_x = std::min(bb.min_x, p.x);
      bb.max_x = std::max(bb.max_x, p.x);
      bb.min_y = std::min(bb.min_y, p.y);
      bb.max_y = std::max(bb.max_y, p.y);
    }
  }

  if (!sampled_graph->points.empty()) {
    sampled_graph->row_index.push_back(sampled_graph->points.size());
  }

  VLOG(1) << "Edges are sampled, chains: " << sampled_graph->chains.size()
          << " points: " << sampled_graph->points.size()
          << " edges: " << n_edges;
  return sampled_graph;
}

}  // namespace rayjoin
#endif  // RAYJOIN_MAP_PLANAR_GRAPH_H
