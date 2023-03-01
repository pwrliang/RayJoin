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
  double a, b, c;  // ax + by + c=0; b >= 0

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
  using point_t = typename cuda_vec<coord_t>::type_2d;
  using edge_t = dev::Edge<COEFFICIENT_T>;

  Map() = default;

  DEV_HOST Map(char id,ArrayView<point_t> points, ArrayView<edge_t> edges)
      : id_(id), points_(points), edges_(edges) {}

  DEV_HOST_INLINE int get_id() const {
      return id_;
  }

  DEV_INLINE const point_t& get_point(index_t offset) const {
    return points_[offset];
  }

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
  using point_t = typename cuda_vec<coord_t>::type_2d;
  using bounding_box_t = BoundingBox<COORD_T>;
  using edge_t = dev::Edge<COEFFICIENT_T>;
  using dev_map_t = dev::Map<COORD_T, COEFFICIENT_T>;

  explicit Map(int id) : id_(id) {}

  template <typename SRC_COORD_T>
  void LoadFrom(
      Stream& stream, const Scaling<SRC_COORD_T, COORD_T>& scaling,
      const pinned_vector<typename cuda_vec<SRC_COORD_T>::type_2d>& points,
      const pinned_vector<edge_t>& edges) {
    points_.resize(points.size());
    // copy to device and calculate edge equations
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
    std::vector<edge_t> edges;

    edges.reserve(pgraph.row_index.size());
    points_.resize(pgraph.points.size());

    ArrayView<typename PlanarGraph<SRC_COORD_T>::point_t> src_points(
        pgraph.points);  // src_points is pinned memory
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

    // split chains into edges
    for (size_t ichain = 0; ichain < pgraph.chains.size(); ichain++) {
      const auto& chain = pgraph.chains[ichain];

      for (auto p_idx = pgraph.row_index[ichain];
           p_idx < pgraph.row_index[ichain + 1] - 1; p_idx++) {
        edge_t e;
        e.eid = p_idx - ichain;  // n points n-1 edges
        e.p1_idx = p_idx;
        e.p2_idx = p_idx + 1;
        e.left_polygon_id = chain.left_polygon_id;
        e.right_polygon_id = chain.right_polygon_id;

        if (pgraph.points[e.p1_idx].x == pgraph.points[e.p2_idx].x &&
            pgraph.points[e.p1_idx].y == pgraph.points[e.p2_idx].y) {
          LOG(FATAL) << "Map " << id_ << " Chain " << chain.id << " lpol "
                     << e.left_polygon_id << ", rpol " << e.right_polygon_id
                     << ", edge " << p_idx - 1 << " has zero-len edge.";
        }
        edges.push_back(e);
      }
    }

    // copy to device and calculate edge equations
    edges_ = edges;
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

  dev_map_t DeviceObject() const {
    return dev_map_t(id_, ArrayView<point_t>(points_), ArrayView<edge_t>(edges_));
  }

  char get_id() const { return id_; }

  size_t get_points_num() const { return points_.size(); }

  size_t get_edges_num() const { return edges_.size(); }

  bounding_box_t bounding_box() const { return bounding_box_; }

 private:
  char id_;
  thrust::device_vector<point_t> points_;
  thrust::device_vector<edge_t> edges_;
  bounding_box_t bounding_box_;
};

}  // namespace rayjoin
#endif  // RAYJOIN_MAP_MAP_H
