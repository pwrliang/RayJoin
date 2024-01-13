#ifndef ALGO_LSI_H
#define ALGO_LSI_H
#include "config.h"
#include "util/rational.h"
#include "util/util.h"

namespace rayjoin {
namespace dev {
template <typename COORD_T>
struct ExactPoint {
  tcb::rational<COORD_T> x, y;

  ExactPoint() = default;

  DEV_HOST ExactPoint(const tcb::rational<COORD_T>& xx,
                      const tcb::rational<COORD_T>& yy)
      : x(xx), y(yy) {}
};

template <typename COORD_T>
struct Intersection : public ExactPoint<COORD_T> {
  index_t eid[2];
  polygon_id_t mid_point_polygon_id;
  DEV_HOST Intersection() : eid{0, 0}, mid_point_polygon_id(DONTKNOW) {}
};

template <typename EDGE_T1, typename EDGE_T2, typename POINT_T,
          typename COEFFICIENT_T>
DEV_INLINE bool intersect_test(const EDGE_T1& e1, const POINT_T& e1_p1,
                               const POINT_T& e1_p2, const EDGE_T2& e2,
                               const POINT_T& e2_p1, const POINT_T& e2_p2) {
#define SUBEDGE(p, e) \
  ((COEFFICIENT_T) p.x * e.a + (COEFFICIENT_T) p.y * e.b + e.c)  // ax+by+c
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
    return false;            // zero length edge
  }

  if (e1_p2_agst_e2 == 0) {
    e1_p2_agst_e2 = -e2.a;
  }
  if (e1_p2_agst_e2 == 0) {
    e1_p2_agst_e2 = -e2.b;
  }
  if (e1_p2_agst_e2 == 0) {
    return false;
  }

  // p1 and p2 of edge1 is on the same side of edge2, they will not
  // intersect
  if ((e1_p1_agst_e2 > 0 && e1_p2_agst_e2 > 0) ||
      (e1_p1_agst_e2 < 0 && e1_p2_agst_e2 < 0)) {
    return false;
  }

  // e2_p1 is on e1
  if (e2_p1_agst_e1 == 0) {
    e2_p1_agst_e1 = e1.a;
  }
  if (e2_p1_agst_e1 == 0) {
    e2_p1_agst_e1 = e1.b;
  }
  if (e2_p1_agst_e1 == 0) {
    return false;
  }
  if (e2_p2_agst_e1 == 0) {
    e2_p2_agst_e1 = e1.a;
  }
  if (e2_p2_agst_e1 == 0) {
    e2_p2_agst_e1 = e1.b;
  }
  if (e2_p2_agst_e1 == 0) {
    return false;
  }
  if ((e2_p1_agst_e1 > 0 && e2_p2_agst_e1 > 0) ||
      (e2_p1_agst_e1 < 0 && e2_p2_agst_e1 < 0)) {
    return false;
  }

  /*
   * Check if both edges are the same.  If so, they shouldn't be
   * intersecting.
   */
  if ((e1_p1 == e2_p1 && e1_p2 == e2_p2) ||
      (e1_p1 == e2_p2 && e1_p2 == e2_p1)) {
    return false;
  }

  return true;
}

template <typename EDGE_T1, typename EDGE_T2, typename POINT_T,
          typename COEFFICIENT_T>
DEV_INLINE bool intersect_test(const EDGE_T1& e1, const POINT_T& e1_p1,
                               const POINT_T& e1_p2, const EDGE_T2& e2,
                               const POINT_T& e2_p1, const POINT_T& e2_p2,
                               tcb::rational<COEFFICIENT_T>& xsect_x,
                               tcb::rational<COEFFICIENT_T>& xsect_y) {
  if (!intersect_test<EDGE_T1, EDGE_T2, POINT_T, COEFFICIENT_T>(
          e1, e1_p1, e1_p2, e2, e2_p1, e2_p2)) {
    return false;
  }
  // calculate intersection point
  auto denom = (COEFFICIENT_T) e1.a * e2.b - (COEFFICIENT_T) e2.a * e1.b;
  auto numx = (COEFFICIENT_T) e2.c * e1.b - (COEFFICIENT_T) e1.c * e2.b;
  auto numy = (COEFFICIENT_T) e2.a * e1.c - (COEFFICIENT_T) e1.a * e2.c;

  xsect_x = tcb::rational<COEFFICIENT_T>(numx, denom);
  xsect_y = tcb::rational<COEFFICIENT_T>(numy, denom);

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
  return true;
}

}  // namespace dev
}  // namespace rayjoin

#endif  // ALGO_LSI_H
