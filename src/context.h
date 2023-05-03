#ifndef RAYJOIN_CONTEXT_H
#define RAYJOIN_CONTEXT_H

#include <libgen.h>
#include <linux/limits.h>
#include <unistd.h>

#include <memory>

#include "map/map.h"
#include "map/planar_graph.h"
#include "map/scaling.h"

namespace rayjoin {

template <typename COORD_T, typename COEFFICIENT_T = __int128>
class Context {
 public:
  using coord_t = COORD_T;
  using coefficient_t = COEFFICIENT_T;
  using scaling_t = Scaling<coord_t>;
  using internal_coord_t = typename scaling_t::internal_coord_t;
  using planar_graph_t = PlanarGraph<coord_t>;
  using map_t = Map<internal_coord_t, coefficient_t>;
  using bounding_box_t = BoundingBox<COORD_T>;

  Context() = default;
  Context(const map_t&) = delete;
  Context& operator=(const map_t&) = delete;

  explicit Context(const std::shared_ptr<planar_graph_t>& pgraph)
      : Context(std::array<std::shared_ptr<planar_graph_t>, 2>{pgraph}) {}

  explicit Context(
      const std::array<std::shared_ptr<planar_graph_t>, 2>& planar_graphs)
      : planar_graphs_(planar_graphs) {
    for (auto pgraph : planar_graphs) {
      if (pgraph != nullptr) {
        auto& bb = pgraph->bb;
        bb_.min_x = std::min(bb_.min_x, bb.min_x);
        bb_.max_x = std::max(bb_.max_x, bb.max_x);
        bb_.min_y = std::min(bb_.min_y, bb.min_y);
        bb_.max_y = std::max(bb_.max_y, bb.max_y);
      }
    }

    scaling_ = Scaling<coord_t>(bb_);
#ifndef NDEBUG
    scaling_.DebugPrint();
#endif
    LOG(INFO) << "Bounding Box, Bottom-left: (" << bb_.min_x << ", "
              << bb_.min_y << "), Top-right: (" << bb_.max_x << ", "
              << bb_.max_y << ")";
    auto internal_min_x = scaling_.ScaleX(bb_.min_x);
    auto internal_min_y = scaling_.ScaleY(bb_.min_y);
    auto internal_max_x = scaling_.ScaleX(bb_.max_x);
    auto internal_max_y = scaling_.ScaleY(bb_.max_y);
    LOG(INFO) << "Scaled Bounding Box, Bottom-left: (" << internal_min_x << ", "
              << internal_min_y << "), Top-right: (" << internal_max_x << ", "
              << internal_max_y << ")";
    LOG(INFO) << "Unscaled Bounding Box, Bottom-left: ("
              << scaling_.UnscaleX(internal_min_x) << ", "
              << scaling_.UnscaleY(internal_min_y) << "), Top-right: ("
              << scaling_.UnscaleX(internal_max_x) << ", "
              << scaling_.UnscaleY(internal_max_y) << ")";

    for (size_t im = 0; im < planar_graphs.size(); im++) {
      auto pgraph = planar_graphs[im];

      if (pgraph != nullptr) {
        auto map = std::make_shared<map_t>(im);

        assert(pgraph != nullptr);
        map->template LoadFrom(stream_, scaling_, *pgraph);
        maps_[im] = map;
      }
    }

    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    const char* path;
    if (count != -1) {
      path = dirname(result);
    }
    exec_root = std::string(path);
  }

  void set_query_map(std::shared_ptr<map_t> query_map) { maps_[1] = query_map; }

  std::shared_ptr<map_t> get_map(int mapno) { return maps_[mapno]; }

  std::shared_ptr<const map_t> get_map(int mapno) const {
    return maps_.at(mapno);
  }

  std::shared_ptr<planar_graph_t> get_planar_graph(int mapno) {
    return planar_graphs_[mapno];
  }

  std::shared_ptr<const planar_graph_t> get_planar_graph(int mapno) const {
    return planar_graphs_.at(mapno);
  }

  size_t get_maps_num() const { return maps_.size(); }

  const scaling_t& get_scaling() const { return scaling_; }

  const bounding_box_t& get_bounding_box() const { return bb_; }

  Stream& get_stream() { return stream_; }

  const std::string& get_exec_root() const { return exec_root; }

 private:
  Stream stream_;
  std::array<std::shared_ptr<planar_graph_t>, 2> planar_graphs_;
  std::array<std::shared_ptr<map_t>, 2> maps_;

  bounding_box_t bb_;
  scaling_t scaling_;
  std::string exec_root;  // folder of binary
};

}  // namespace rayjoin

#endif  // RAYJOIN_CONTEXT_H
