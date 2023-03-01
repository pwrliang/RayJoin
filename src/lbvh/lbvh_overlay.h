#ifndef RAYJOIN_LBVH_OVERLAY_H
#define RAYJOIN_LBVH_OVERLAY_H
#include "algo/lsi.h"
#include "context.h"
#include "lbvh.cuh"

namespace rayjoin {

template <typename CONTEXT_T>
class LBVHOverlay {
  using coord_t = typename CONTEXT_T::coord_t;
  using internal_coord_t = typename CONTEXT_T::internal_coord_t;
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using xsect_t = dev::Intersection<internal_coord_t>;

 public:
  struct segment {
    float2 p1, p2;
  };

  struct aabb_getter {
    __device__ lbvh::aabb<float> operator()(const segment& f) const noexcept {
      lbvh::aabb<float> box;
      auto min_x = min(f.p1.x, f.p2.x);
      auto max_x = max(f.p1.x, f.p2.x);
      auto min_y = min(f.p1.y, f.p2.y);
      auto max_y = max(f.p1.y, f.p2.y);

      box.lower = make_float4(min_x, min_y, 0, 0);
      box.upper = make_float4(max_x, max_y, 0, 0);
      return box;
    }
  };

  explicit LBVHOverlay(CONTEXT_T& ctx, float xsect_factor = 10) : ctx_(ctx) {
    size_t total_n_edges = 0;

    FOR2 {
      auto n_edges = ctx.get_map(im)->get_edges_num();

      total_n_edges += n_edges;
      max_n_xsects_ += n_edges * xsect_factor;
    }
  }

  void Init() {
    auto scaling = ctx_.get_scaling();
    auto bb = ctx_.get_bounding_box();
    auto mid_x = (bb.min_x + bb.max_x) / 2.0;
    auto margin =
        scaling.ScaleX(mid_x + PRIMITIVE_EPSILON_X) - scaling.ScaleX(mid_x);
    auto& stream = ctx_.get_stream();

    xsect_edges_.Init(max_n_xsects_);

    FOR2 {
      auto map = this->ctx_.get_map(im);
      auto d_map = map->DeviceObject();
      auto points_num = map->get_points_num();
      point_in_polygon_[im].resize(points_num, -1);

      primitives_[im].resize(map->get_edges_num());

      ForEach(
          stream, map->get_edges_num(),
          [=] __device__(size_t eid, ArrayView<segment> edges) {
            const auto& e = d_map.get_edge(eid);
            auto p1 = d_map.get_point(e.p1_idx);
            auto p2 = d_map.get_point(e.p2_idx);
            if (p1.x > p2.x) {
              SWAP(p1, p2);
            }
            auto x1 = scaling.UnscaleX(p1.x);
            auto y1 = scaling.UnscaleY(p1.y);
            auto x2 = scaling.UnscaleX(p2.x);
            auto y2 = scaling.UnscaleY(p2.y);

            //            if (e.b != 0) {
            //              assert(p1.x < p2.x);
            //              // use double is much faster than rational
            //              // this does not need to be accurate
            //              double a = -e.a / e.b;
            //              double b = -e.c / e.b;
            //
            //              auto new_x1 = p1.x - margin;
            //              auto new_y1 = a * new_x1 + b;
            //
            //              auto new_x2 = p2.x + margin;
            //              auto new_y2 = a * new_x2 + b;
            //
            //              x1 = scaling.UnscaleX(new_x1);
            //              y1 = scaling.UnscaleY(new_y1);
            //              x2 = scaling.UnscaleX(new_x2);
            //              y2 = scaling.UnscaleY(new_y2);
            //            }

            edges[eid].p1 = {(float) x1, (float) y1};
            edges[eid].p2 = {(float) x2, (float) y2};
          },
          ArrayView<segment>(primitives_[im]));
      stream.Sync();
    }
  }

  void AddMapToGrid() {
    FOR2 {
      bvh_[im] = lbvh::bvh<float, segment, aabb_getter>(primitives_[im]);

      primitives_[im].resize(0);
      primitives_[im].shrink_to_fit();
    }
  }

  void IntersectEdge() {
    int src_map_id = 0;
    int dst_map_id = 1 - src_map_id;
    auto d_src_map = ctx_.get_map(src_map_id)->DeviceObject();
    auto d_dst_map = ctx_.get_map(dst_map_id)->DeviceObject();
    auto scaling = ctx_.get_scaling();
    auto& stream = ctx_.get_stream();
    const auto bvh_dev = bvh_[dst_map_id].get_device_repr();
    auto d_xsects = xsect_edges_.DeviceObject();

    xsect_edges_.Clear(stream);
#ifndef NDEBUG
    SharedValue<uint64_t> total_tests;
    auto d_total_tests = total_tests.data();
    total_tests.set(0, stream);
#endif

    ForEach(
        stream, d_src_map.get_edges_num(), [=] __device__(size_t eid1) mutable {
          const auto& e1 = d_src_map.get_edge(eid1);
          const auto& e1_p1 = d_src_map.get_point(e1.p1_idx);
          const auto& e1_p2 = d_src_map.get_point(e1.p2_idx);
          auto x1 = scaling.UnscaleX(e1_p1.x);
          auto y1 = scaling.UnscaleY(e1_p1.y);
          auto x2 = scaling.UnscaleX(e1_p2.x);
          auto y2 = scaling.UnscaleY(e1_p2.y);

          auto min_x = min(x1, x2);
          auto max_x = max(x1, x2);
          auto min_y = min(y1, y2);
          auto max_y = max(y1, y2);

          lbvh::aabb<float> box;
          box.upper = make_float4(max_x, max_y, 0.0, 0.0);
          box.lower = make_float4(min_x, min_y, 0.0, 0.0);

          auto pair = lbvh::query_device_all(
              bvh_dev, lbvh::overlaps(box),
              [=] __device__(std::uint32_t eid2) mutable {
                const auto& e2 = d_dst_map.get_edge(eid2);
                const auto& e2_p1 = d_dst_map.get_point(e2.p1_idx);
                const auto& e2_p2 = d_dst_map.get_point(e2.p2_idx);

#define SUBEDGE(p, e) \
  ((coefficient_t) p.x * e.a + (coefficient_t) p.y * e.b + e.c)  // ax+by+c
                // N.B., e.b >= 0, we ensure it when calculate edge eqns
                auto e2_p1_agst_e1 = SUBEDGE(e2_p1, e1);
                auto e2_p2_agst_e1 = SUBEDGE(e2_p2, e1);
                auto e1_p1_agst_e2 = SUBEDGE(e1_p1, e2);
                auto e1_p2_agst_e2 = SUBEDGE(e1_p2, e2);
#undef SUBEDGE
                // e1_p1 is on e2
                if (e1_p1_agst_e2 == 0) {
                  e1_p1_agst_e2 = -e2.a;
                }
                if (e1_p1_agst_e2 == 0) {  // a = 0, e2 is parallel to x-axis
                  e1_p1_agst_e2 = -e2.b;
                }
                if (e1_p1_agst_e2 == 0) {  // b = 0, then c must be 0
                  return;                  // zero length edge
                }

                if (e1_p2_agst_e2 == 0) {
                  e1_p2_agst_e2 = -e2.a;
                }
                if (e1_p2_agst_e2 == 0) {
                  e1_p2_agst_e2 = -e2.b;
                }
                if (e1_p2_agst_e2 == 0) {
                  return;
                }

                // p1 and p2 of edge1 is on the same side of edge2, they will
                // not intersect
                if ((e1_p1_agst_e2 > 0 && e1_p2_agst_e2 > 0) ||
                    (e1_p1_agst_e2 < 0 && e1_p2_agst_e2 < 0)) {
                  return;
                }

                // e2_p1 is on e1
                if (e2_p1_agst_e1 == 0) {
                  e2_p1_agst_e1 = e1.a;
                }
                if (e2_p1_agst_e1 == 0) {
                  e2_p1_agst_e1 = e1.b;
                }
                if (e2_p1_agst_e1 == 0) {
                  return;
                }
                if (e2_p2_agst_e1 == 0) {
                  e2_p2_agst_e1 = e1.a;
                }
                if (e2_p2_agst_e1 == 0) {
                  e2_p2_agst_e1 = e1.b;
                }
                if (e2_p2_agst_e1 == 0) {
                  return;
                }
                if ((e2_p1_agst_e1 > 0 && e2_p2_agst_e1 > 0) ||
                    (e2_p1_agst_e1 < 0 && e2_p2_agst_e1 < 0)) {
                  return;
                }

                /*
                 * Check if both edges are the same.  If so, they shouldn't be
                 * intersecting.
                 */
                if ((e1_p1 == e2_p1 && e1_p2 == e2_p2) ||
                    (e1_p1 == e2_p2 && e1_p2 == e2_p1)) {
                  return;
                }

                auto denom =
                    (coefficient_t) e1.a * e2.b - (coefficient_t) e2.a * e1.b;
                auto numx =
                    (coefficient_t) e2.c * e1.b - (coefficient_t) e1.c * e2.b;
                auto numy =
                    (coefficient_t) e2.a * e1.c - (coefficient_t) e1.a * e2.c;

                tcb::rational<coefficient_t> xsect_x(numx, denom);
                tcb::rational<coefficient_t> xsect_y(numy, denom);

                auto t = MIN4(e1_p1.x, e1_p2.x, e2_p1.x, e2_p2.x);
                if (xsect_x < t) {
                  xsect_x = t;
                }

                t = MAX4(e1_p1.x, e1_p2.x, e2_p1.x, e2_p2.x);
                if (xsect_x > t) {
                  xsect_x = t;
                }

                t = MIN4(e1_p1.y, e1_p2.y, e2_p1.y, e2_p2.y);
                if (xsect_y < t) {
                  xsect_y = t;
                }
                t = MAX4(e1_p1.y, e1_p2.y, e2_p1.y, e2_p2.y);
                if (xsect_y > t) {
                  xsect_y = t;
                }

                dev::Intersection<internal_coord_t> xsect;
                xsect.x = xsect_x;
                xsect.y = xsect_y;
                xsect.eid[0] = eid1;
                xsect.eid[1] = eid2;
                d_xsects.AppendWarp(xsect);
              });
#ifndef NDEBUG
          auto traversed_aabbs = pair.first;
          auto num_found = pair.second;
          atomicAdd(reinterpret_cast<unsigned long long int*>(d_total_tests),
                    num_found);
#endif
        });

    size_t n_xsects = xsect_edges_.size(stream);
    stream.Sync();
    LOG(INFO) << "# of xsects: " << n_xsects;
#ifndef NDEBUG
    LOG(INFO) << "Total Tests: " << total_tests.get(stream);
#endif
  }

  void LocateVerticesInOtherMap(int src_map_id) {
    auto& ctx = this->ctx_;
    auto& scaling = ctx.get_scaling();
    auto& stream = ctx.get_stream();
    int dst_map_id = 1 - src_map_id;
    auto d_src_map = ctx.get_map(src_map_id)->DeviceObject();
    auto d_dst_map = ctx.get_map(dst_map_id)->DeviceObject();
    ArrayView<polygon_id_t> d_point_in_polygon(point_in_polygon_[src_map_id]);
    const auto bvh_dev = bvh_[dst_map_id].get_device_repr();

    ForEach(
        stream, d_src_map.get_points_num(),
        [=] __device__(size_t point_idx) mutable {
          for (auto point_idx = TID_1D; point_idx < d_src_map.get_points_num();
               point_idx += TOTAL_THREADS_1D) {
            polygon_id_t ipol = 0;  // if not hit any edges, vertex is on face0
            const auto& src_p = d_src_map.get_point(point_idx);
            auto x = scaling.UnscaleX(src_p.x);
            auto y = scaling.UnscaleY(src_p.y);

            lbvh::aabb<float> box;

            box.upper = make_float4(x, y + RAY_TMAX, 0, 0);
            box.lower = make_float4(x, y, 0, 0);
            auto best_y = std::numeric_limits<internal_coord_t>::max();
            internal_coord_t best_y_max;
            const dev::Edge<coefficient_t>* best_e = nullptr;

            lbvh::query_device_all(
                bvh_dev, lbvh::overlaps(box),
                [&] __device__(std::uint32_t eid) mutable {
                  const auto& e = d_dst_map.get_edge(eid);
                  auto& p1 = d_dst_map.get_point(e.p1_idx);
                  auto& p2 = d_dst_map.get_point(e.p2_idx);
                  auto x_min = min(p1.x, p2.x);
                  auto x_max = max(p1.x, p2.x);
                  auto x_src_p = src_p.x;
                  auto y_src_p = src_p.y;
                  coefficient_t e_a = e.a, e_b = e.b, e_c = e.c;

                  /*
                   * Is point outside x bounds of this edge?  Use simulation of
                   * simplicity: shift map 1 by epsilon relative to map 0.  This
                   * also eliminates vertical edges.
                   */
                  if (x_src_p < x_min || x_src_p > x_max ||
                      x_src_p == ((src_map_id == 0) ? x_min : x_max)) {
                    return;
                  }

                  assert(e.b != 0);
                  /*
                   * Is point above edge?  If point is on edge, use simulation
                   * of simplicity.
                   */
                  tcb::rational<coefficient_t> xsect_y(-e_a * x_src_p - e_c,
                                                       e_b);
                  auto diff_y = y_src_p - xsect_y;

                  if (diff_y == 0) {
                    diff_y = (src_map_id == 0 ? -e_a : e_a);
                  }
                  if (diff_y == 0) {
                    diff_y = (src_map_id == 0 ? -e_b : e_b);
                  }
#ifndef NDEBUG
                  if (diff_y == 0) {
                    printf("Zero length edge\n");
                  }
#endif
                  // current point is above the current edge
                  if (diff_y > 0) {
                    return;
                  }
                  if (xsect_y > best_y) {
                    return;
                  }
                  if (xsect_y ==
                      best_y) { /* Line up from point cuts this edge and another
                                 * edge at the same Y.  This should only happen
                                 * at a vertex.  Roundoff error should not cause
                                 * it, I hope.  If im==0, we're at the right end
                                 * of both edges.  im==1 => left end.  */
                    //      printf("Tie\n");
                    /* flag:  Is this slope algebraically less than the last
                     * one? */

                    bool flag =
                        ((double) e_a / e_b) > ((double) best_e->a / best_e->b);

                    /* If im==0 we want the bigger slope, if im==1, the smaller.
                     */

                    if ((src_map_id && !flag) || (flag && !src_map_id))
                      return;
                  }
                  best_y = xsect_y;
                  best_e = &e;
                  best_y_max = max(p1.y, p2.y);
                });
            if (best_e != nullptr) {
              if (d_dst_map.get_point(best_e->p1_idx).x <
                  d_dst_map.get_point(best_e->p2_idx).x) {
                ipol = best_e->right_polygon_id;
              } else {
                ipol = best_e->left_polygon_id;
              }
              break;
            }
            d_point_in_polygon[point_idx] = ipol;
          }
        });
    stream.Sync();
  }

  const thrust::device_vector<polygon_id_t>& get_point_in_polygon(
      int im) const {
    return point_in_polygon_[im];
  }

  const thrust::device_vector<xsect_t>& get_xsect_edges(int im) const {
    return xsect_edges_sorted_[im];
  }

 protected:
  CONTEXT_T& ctx_;
  uint32_t max_n_xsects_{};
  thrust::device_vector<segment> primitives_[2];
  Queue<xsect_t> xsect_edges_;
  thrust::device_vector<xsect_t> xsect_edges_sorted_[2];
  thrust::device_vector<polygon_id_t> point_in_polygon_[2];

  lbvh::bvh<float, segment, aabb_getter> bvh_[2];
};

}  // namespace rayjoin
#endif  // RAYJOIN_LBVH_OVERLAY_H
