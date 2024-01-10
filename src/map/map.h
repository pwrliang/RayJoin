#ifndef RAYJOIN_MAP_MAP_H
#define RAYJOIN_MAP_MAP_H
#include <thrust/device_vector.h>

#include "config.h"
#include "map/bounding_box.h"
#include "map/planar_graph.h"
#include "map/scaling.h"
#include "util/array_view.h"
#include "util/derived_atomic_functions.h"
#include "util/launcher.h"
#include "util/shared_value.h"
#include "util/stream.h"
#include "util/type_traits.h"

namespace rayjoin {
namespace dev {

template <typename COEFFICIENT_T>
struct EdgeEquation {
  coefficient_t a, b, c;  // ax + by + c=0; b >= 0

  EdgeEquation() = default;

  template <typename POINT_T>
  DEV_HOST EdgeEquation(const POINT_T& p1, const POINT_T& p2) {
    a = p1.y - p2.y;
    b = p2.x - p1.x;
    c = -(COEFFICIENT_T) p1.x * a - (COEFFICIENT_T) p1.y * b;

    assert(a != 0 || b != 0);

    if (b < 0) {
      a = -a;
      b = -b;
      c = -c;
    }
  }
};

template <typename COEFFICIENT_T>
struct __builtin_align__(16) Edge : public EdgeEquation<COEFFICIENT_T> {
  index_t eid;
  index_t p1_idx, p2_idx;
  index_t left_polygon_id, right_polygon_id;
};

template <typename COORD_T, typename COEFFICIENT_T>
class Map {
 public:
  using coord_t = COORD_T;
  using coefficient_t = COEFFICIENT_T;
  using point_t = typename cuda_vec<coord_t>::type_2d;
  using edge_t = dev::Edge<COEFFICIENT_T>;

  Map() = default;

  DEV_HOST Map(char id, ArrayView<point_t> points, ArrayView<edge_t> edges)
      : id_(id), points_(points), edges_(edges) {}

  DEV_HOST_INLINE int get_id() const { return id_; }

  DEV_HOST_INLINE size_t get_points_num() const { return points_.size(); }

  DEV_HOST_INLINE size_t get_edges_num() const { return edges_.size(); }

  DEV_INLINE const point_t& get_point(size_t point_idx) const {
    return points_[point_idx];
  }

  DEV_INLINE const edge_t& get_edge(size_t edge_idx) const {
    return edges_[edge_idx];
  }

  DEV_HOST_INLINE ArrayView<point_t> get_points() const { return points_; }

  DEV_HOST_INLINE ArrayView<edge_t> get_edges() const { return edges_; }

  DEV_HOST_INLINE polygon_id_t get_face_id(const edge_t& e) const {
    polygon_id_t ipol;
    if (get_point(e.p1_idx).x < get_point(e.p2_idx).x) {
      ipol = e.right_polygon_id;
    } else {
      ipol = e.left_polygon_id;
    }
    return ipol;
  }

 private:
  char id_;
  ArrayView<point_t> points_;
  ArrayView<edge_t> edges_;
};

}  // namespace dev

template <typename COORD_T, typename COEFFICIENT_T>
class Map {
 public:
  using coord_t = COORD_T;
  using coefficient_t = COEFFICIENT_T;
  using point_t = typename cuda_vec<coord_t>::type_2d;
  using edge_t = dev::Edge<COEFFICIENT_T>;
  using dev_map_t = dev::Map<COORD_T, COEFFICIENT_T>;

  explicit Map(int id) : id_(id) {}

  template <typename SRC_COORD_T>
  void LoadFrom(
      Stream& stream, const Scaling<SRC_COORD_T, COORD_T>& scaling,
      const pinned_vector<typename cuda_vec<SRC_COORD_T>::type_2d>& points,
      const pinned_vector<edge_t>& edges) {
    points_.resize(points.size());
    edges_ = edges;

    ArrayView<typename PlanarGraph<SRC_COORD_T>::point_t> src_points(
        points);  // src_points is pinned memory
    ArrayView<point_t> dst_points(points_);

    ForEach(stream, src_points.size(),
            [=] __device__(size_t point_idx) mutable {
              auto x = src_points[point_idx].x;
              auto y = src_points[point_idx].y;

              dst_points[point_idx].x = scaling.ScaleX(x);
              dst_points[point_idx].y = scaling.ScaleY(y);
            });

    // copy to device and calculate edge equations
    LaunchKernel(
        stream,
        [=] __device__(ArrayView<point_t> points, ArrayView<edge_t> edges) {
          for (unsigned int eid = TID_1D; eid < edges.size();
               eid += TOTAL_THREADS_1D) {
            auto& e = edges[eid];
            const auto& p1 = points[e.p1_idx];
            const auto& p2 = points[e.p2_idx];
            auto x1 = p1.x;
            auto y1 = p1.y;
            auto x2 = p2.x;
            auto y2 = p2.y;

            e.a = y1 - y2;
            e.b = x2 - x1;
            e.c = -(COEFFICIENT_T) x1 * e.a - (COEFFICIENT_T) y1 * e.b;

            assert(e.a != 0 || e.b != 0);

            if (e.b < 0) {
              e.a = -e.a;
              e.b = -e.b;
              e.c = -e.c;
            }
          }
        },
        ArrayView<point_t>(points_), ArrayView<edge_t>(edges_));

    stream.Sync();
  }

  template <typename SRC_COORD_T>
  void LoadFrom(Stream& stream, const Scaling<SRC_COORD_T, COORD_T>& scaling,
                const PlanarGraph<SRC_COORD_T>& pgraph) {
    points_.resize(pgraph.points.size());
    edges_.resize(pgraph.points.size() - pgraph.chains.size());

    ArrayView<typename PlanarGraph<SRC_COORD_T>::point_t> src_points(
        pgraph.points);
    ArrayView<point_t> dst_points(points_);

    LaunchKernel(stream, [=] __device__() mutable {
      for (auto point_idx = TID_1D; point_idx < src_points.size();
           point_idx += TOTAL_THREADS_1D) {
        auto x = src_points[point_idx].x;
        auto y = src_points[point_idx].y;

        dst_points[point_idx].x = scaling.ScaleX(x);
        dst_points[point_idx].y = scaling.ScaleY(y);
      }
    });

    // copy to device and calculate edge equations

    ArrayView<Chain> v_chains(pgraph.chains);
    ArrayView<index_t> v_row_index(pgraph.row_index);

    LaunchKernel(
        stream,
        [=] __device__(ArrayView<point_t> points, ArrayView<edge_t> edges) {
          auto warp_id = TID_1D / 32;
          auto n_warps = TOTAL_THREADS_1D / 32;
          auto lane_id = threadIdx.x % 32;

          for (size_t ichain = warp_id; ichain < v_chains.size();
               ichain += n_warps) {
            const auto& chain = v_chains[ichain];

            for (auto p_idx = v_row_index[ichain] + lane_id;
                 p_idx < v_row_index[ichain + 1] - 1; p_idx += 32) {
              auto eid = p_idx - ichain;  // n points n-1 edges
              auto& e = edges[eid];

              e.eid = p_idx - ichain;  // n points n-1 edges
              e.p1_idx = p_idx;
              e.p2_idx = p_idx + 1;
              e.left_polygon_id = chain.left_polygon_id;
              e.right_polygon_id = chain.right_polygon_id;

              const auto& p1 = points[e.p1_idx];
              const auto& p2 = points[e.p2_idx];
              auto x1 = p1.x;
              auto y1 = p1.y;
              auto x2 = p2.x;
              auto y2 = p2.y;

              e.a = y1 - y2;
              e.b = x2 - x1;
              e.c = -(COEFFICIENT_T) x1 * e.a - (COEFFICIENT_T) y1 * e.b;

              assert(e.a != 0 || e.b != 0);

              if (e.b < 0) {
                e.a = -e.a;
                e.b = -e.b;
                e.c = -e.c;
              }
            }
          }
        },
        ArrayView<point_t>(points_), ArrayView<edge_t>(edges_));

    stream.Sync();
  }

  /**
   * Copy points and edges from device to host for debugging only
   */
  void D2H() {
    if (h_points_.empty()) {
      h_points_ = points_;
    }
    if (h_edges_.empty()) {
      h_edges_ = edges_;
    }
  }

  dev_map_t DeviceObject() const {
    return dev_map_t(id_, ArrayView<point_t>(points_),
                     ArrayView<edge_t>(edges_));
  }

  char get_id() const { return id_; }

  size_t get_points_num() const { return points_.size(); }

  size_t get_edges_num() const { return edges_.size(); }

  const point_t& get_point(size_t point_idx) const {
    assert(point_idx < h_points_.size());
    return h_points_[point_idx];
  }

  const edge_t& get_edge(size_t edge_idx) const {
    assert(edge_idx < h_edges_.size());
    return h_edges_[edge_idx];
  }

  polygon_id_t get_face_id(const edge_t& e) const {
    polygon_id_t ipol;
    if (get_point(e.p1_idx).x < get_point(e.p2_idx).x) {
      ipol = e.right_polygon_id;
    } else {
      ipol = e.left_polygon_id;
    }
    return ipol;
  }

  std::string ScaledEndpointsToString(size_t eid) const {
    return ScaledEndpointsToString(get_edge(eid));
  }

  std::string ScaledEndpointsToString(const edge_t& e) const {
    auto& p1 = get_point(e.p1_idx);
    auto& p2 = get_point(e.p2_idx);

    std::string s;
    s.resize(1024);

    auto n = snprintf(const_cast<char*>(s.c_str()), s.size(),
                      "(%ld, %ld) - (%ld, %ld)", p1.x, p1.y, p2.x, p2.y);

    s.resize(n);

    return s;
  }

  template <typename SCALING_T>
  std::string EndpointsToString(size_t eid, const SCALING_T& scaling) const {
    return EndpointsToString(get_edge(eid), scaling);
  }

  template <typename SCALING_T>
  std::string EndpointsToString(const edge_t& e,
                                const SCALING_T& scaling) const {
    auto p1_idx = e.p1_idx;
    auto p2_idx = e.p2_idx;
    auto p1 = get_point(p1_idx);
    auto p2 = get_point(p2_idx);
    auto x1 = scaling.UnscaleX(p1.x);
    auto y1 = scaling.UnscaleY(p1.y);
    auto x2 = scaling.UnscaleX(p2.x);
    auto y2 = scaling.UnscaleY(p2.y);

    std::string s;
    s.resize(1024);

    auto n = snprintf(const_cast<char*>(s.c_str()), s.size(),
                      "(%.8lf, %.8lf) - (%.8lf, %.8lf)", x1, y1, x2, y2);

    s.resize(n);

    return s;
  }

  bool IsPointOnLine(const edge_t& e, const point_t& p) const {
    auto& p1 = get_point(e.p1_idx);
    auto& p2 = get_point(e.p2_idx);

    return p1.x == p.x && p1.y == p.y || p2.x == p.x && p2.y == p.y;
  }

  const thrust::device_vector<point_t>& get_points() const { return points_; }

 private:
  int id_;
  thrust::device_vector<point_t> points_;
  thrust::device_vector<edge_t> edges_;
  pinned_vector<point_t> h_points_;  // For debugging
  pinned_vector<edge_t> h_edges_;
};

}  // namespace rayjoin
#endif  // RAYJOIN_MAP_MAP_H
