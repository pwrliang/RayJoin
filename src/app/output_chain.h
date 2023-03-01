#ifndef RAYJOIN_APP_OUTPUT_CHAIN_H
#define RAYJOIN_APP_OUTPUT_CHAIN_H
#include <thrust/device_vector.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include "algo/lsi.h"
#include "config.h"
#include "glog/logging.h"
#include "map/scaling.h"
#include "util/type_traits.h"

namespace rayjoin {
template <typename COORD_T>
struct OutputChain {
  int64_t id;
  std::vector<typename cuda_vec<COORD_T>::type_2d> points;
  index_t first_point_idx;
  index_t last_point_idx;
  int64_t left_polygon_id;   // left polygon id of the chain
  int64_t right_polygon_id;  // right polygon id of the chain
  int64_t other_map_polygon_id;

  void AddChainPoint(const typename cuda_vec<COORD_T>::type_2d& p) {
    points.push_back(p);
  }

  template <typename INTERNAL_COORD_T>
  void AddXsectPoint(const dev::Intersection<INTERNAL_COORD_T>& xsect,
                     const Scaling<COORD_T, INTERNAL_COORD_T>& scaling) {
    typename cuda_vec<COORD_T>::type_2d p{scaling.UnscaleX(xsect.x),
                                          scaling.UnscaleY(xsect.y)};
    points.push_back(p);
  }
};

template <typename CONTEXT_T>
void WriteOutputChain(
    CONTEXT_T& ctx,
    const thrust::device_vector<
        dev::Intersection<typename CONTEXT_T::internal_coord_t>>*
        xsect_edges_sorted_pair,
    const thrust::device_vector<polygon_id_t>* point_in_polygon_pair,
    const char* path) {
  using coord_t = typename CONTEXT_T::coord_t;
  using internal_coord_t = typename CONTEXT_T::internal_coord_t;
  using xsect_t = dev::Intersection<internal_coord_t>;

  const auto& scaling = ctx.get_scaling();
  std::vector<OutputChain<coord_t>> output_chains;

  auto flush = [&output_chains](OutputChain<coord_t>& output_chain) {
    auto& points = output_chain.points;

    if (!points.empty()) {
      if (output_chain.left_polygon_id * output_chain.other_map_polygon_id !=
              0 ||
          output_chain.right_polygon_id * output_chain.other_map_polygon_id !=
              0) {
        auto p_it = std::unique(points.begin(), points.end(),
                                [](const double2& a, const double2& b) {
                                  return a.x == b.x && a.y == b.y;
                                });
        points.resize(std::distance(points.begin(), p_it));
        output_chain.id = output_chains.size();
        output_chains.push_back(output_chain);
        points.clear();
      }
      points.clear();
    }
  };

  pinned_vector<xsect_t> xsect_edges_sorted;
  pinned_vector<polygon_id_t> point_in_polyon;

  FOR2 {
    xsect_edges_sorted = xsect_edges_sorted_pair[im];
    point_in_polyon = point_in_polygon_pair[im];
    const auto& p_graph = *ctx.get_planar_graph(im);

    std::unordered_map<index_t, std::vector<xsect_t>> grouped_xsects;

    for (auto& xsect : xsect_edges_sorted) {
      grouped_xsects[xsect.eid[im]].push_back(xsect);
    }
    LOG(INFO) << "Map " << im << ", Xsect: " << xsect_edges_sorted.size() << " "
              << grouped_xsects.size();
    for (size_t ic = 0; ic < p_graph.chains.size(); ic++) {
      const auto& chain = p_graph.chains[ic];
      auto begin_pid = p_graph.row_index[ic];
      auto end_pid = p_graph.row_index[ic + 1];

      OutputChain<coord_t> output_chain;

      output_chain.left_polygon_id = chain.left_polygon_id;
      output_chain.right_polygon_id = chain.right_polygon_id;

      for (auto pid = begin_pid; pid < end_pid; pid++) {
        output_chain.other_map_polygon_id = point_in_polyon[pid];
        output_chain.AddChainPoint(
            p_graph.points[pid]);  // last point in the chain

        if (pid != end_pid - 1) {
          auto eid = pid - ic;
          auto it = grouped_xsects.find(eid);

          // if this edge in the chain intersects with other map
          if (it != grouped_xsects.end()) {
            auto& xsects = it->second;

            if (!xsects.empty()) {
              output_chain.template AddXsectPoint(xsects[0], scaling);

              for (size_t ixsect = 0; ixsect < xsects.size() - 1; ixsect++) {
                const xsect_t& xsect = xsects[ixsect];
                const xsect_t& next_xsect = xsects[ixsect + 1];

                flush(output_chain);

                output_chain.other_map_polygon_id = xsect.mid_point_polygon_id;
                output_chain.template AddXsectPoint(xsect, scaling);
                output_chain.template AddXsectPoint(next_xsect, scaling);
              }

              flush(output_chain);
              // add last xsect
              output_chain.template AddXsectPoint(xsects.back(), scaling);
            }
          }
        }
      }

      flush(output_chain);
    }
  }

  std::map<std::pair<int64_t, int64_t>, size_t> face_ids;
  std::unordered_map<typename cuda_vec<coord_t>::type_2d, index_t> point_ids;
  index_t point_counter = 0;

  auto create_polygon = [&](int64_t polygon_id1,
                            int64_t polygon_id2) -> size_t {
    if (polygon_id1 == 0 || polygon_id2 == 0) {
      return 0;
    }
    auto k = std::make_pair(polygon_id1, polygon_id2);
    auto it = face_ids.find(k);
    if (it == face_ids.end()) {
      face_ids[k] = face_ids.size() + 1;
      return face_ids.size();
    }
    return it->second;
  };

  for (auto& chain : output_chains) {
    if (chain.left_polygon_id < chain.other_map_polygon_id) {
      chain.left_polygon_id =
          create_polygon(chain.left_polygon_id, chain.other_map_polygon_id);
    } else {
      chain.left_polygon_id =
          create_polygon(chain.other_map_polygon_id, chain.left_polygon_id);
    }

    if (chain.right_polygon_id < chain.other_map_polygon_id) {
      chain.right_polygon_id =
          create_polygon(chain.right_polygon_id, chain.other_map_polygon_id);
    } else {
      chain.right_polygon_id =
          create_polygon(chain.other_map_polygon_id, chain.right_polygon_id);
    }

    for (const auto& p : chain.points) {
      if (point_ids.find(p) == point_ids.end()) {
        point_ids[p] = point_counter++;
      }
    }
    chain.first_point_idx = point_ids[chain.points.front()];
    chain.last_point_idx = point_ids[chain.points.back()];
  }

  LOG(INFO) << "Total chains: " << output_chains.size()
            << " Total faces: " << face_ids.size();

  std::ofstream ofs(path);
  CHECK(ofs.is_open()) << "Cannot open " << path;
  ofs.setf(std::ios::fixed, std::ios::floatfield);
  ofs.precision(6);

  for (size_t ichain = 0; ichain < output_chains.size(); ichain++) {
    const auto& chain = output_chains[ichain];
    ofs << (ichain + 1) << " " << chain.points.size() << " "
        << chain.first_point_idx << " " << chain.last_point_idx << " "
        << chain.left_polygon_id << " " << chain.right_polygon_id << '\n';

    for (const auto& p : chain.points) {
      ofs << p.x << " " << p.y << '\n';
    }
  }

  ofs.close();
}

}  // namespace rayjoin
#endif  // RAYJOIN_APP_OUTPUT_CHAIN_H
