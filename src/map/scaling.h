#ifndef RAYJOIN_MAP_SCALING_H
#define RAYJOIN_MAP_SCALING_H
#include <cmath>

#include "map/bounding_box.h"
#include "util/type_traits.h"
#include "util/util.h"

namespace rayjoin {

namespace detail {
template <typename COORD_T>
struct internal_coord {};

template <>
struct internal_coord<float> {
  using type = int32_t;
  constexpr static int shift = 1;
};

template <>
struct internal_coord<double> {
  using type = int64_t;
  constexpr static int shift = 17;
};
}  // namespace detail

template <typename COORD_T,
          typename INTERNAL_COORD_T =
              typename detail::internal_coord<COORD_T>::type,
          int n_shift = detail::internal_coord<COORD_T>::shift>
class Scaling {
  static_assert(std::is_floating_point<COORD_T>::value,
                "COORD_T should be floating point type");
  static_assert(std::is_integral<INTERNAL_COORD_T>::value &&
                    std::is_signed<INTERNAL_COORD_T>::value,
                "INTERNAL_COORD_T should be signed integral type");

 public:
  using coord_t = COORD_T;
  using internal_coord_t = INTERNAL_COORD_T;

  DEV_HOST explicit Scaling()
      : internal_max_(std::numeric_limits<internal_coord_t>::max() >> n_shift),
        internal_min_(std::numeric_limits<internal_coord_t>::min() >> n_shift),
        internal_range_(internal_max_ - internal_min_),
        rx_(0),
        ry_(0),
        rrx_(0),
        rry_(0),
        deltax_(0),
        deltay_(0),
        ddeltax_(0),
        ddeltay_(0) {}

  DEV_HOST explicit Scaling(const BoundingBox<coord_t>& bb) : Scaling() {
    auto max_x = bb.max_x + SCALING_BOUNDING_BOX_MARGIN;
    auto min_x = bb.min_x - SCALING_BOUNDING_BOX_MARGIN;
    auto max_y = bb.max_y + SCALING_BOUNDING_BOX_MARGIN;
    auto min_y = bb.min_y - SCALING_BOUNDING_BOX_MARGIN;

    rx_ = (coord_t) internal_range_ / (max_x - min_x);
    ry_ = (coord_t) internal_range_ / (max_y - min_y);
    rrx_ = 1 / rx_;
    rry_ = 1 / ry_;

    deltax_ = 0.5 * ((internal_max_ + internal_min_) - (max_x + min_x) * rx_);
    deltay_ = 0.5 * ((internal_max_ + internal_min_) - (max_y + min_y) * ry_);
    ddeltax_ = 0.5 * ((max_x + min_x) - (internal_max_ + internal_min_) * rrx_);
    ddeltay_ = 0.5 * ((max_y + min_y) - (internal_max_ + internal_min_) * rry_);
  }

  DEV_HOST void DebugPrint() {
    printf("(rx, ry) = (%lf, %lf)\n", (double) rx_, (double) ry_);
    printf("(deltax, deltay) = (%lf, %lf)\n", (double) deltax_,
           (double) deltay_);
  }

  DEV_HOST_INLINE internal_coord_t ScaleX(coord_t x) const {
    internal_coord_t internal_x = x * rx_ + deltax_;

    assert(internal_x >= internal_min_ && internal_x <= internal_max_);

    return internal_x;
  }

  DEV_HOST_INLINE internal_coord_t ScaleY(coord_t y) const {
    internal_coord_t internal_y = y * ry_ + deltay_;

    assert(internal_y >= internal_min_ && internal_y <= internal_max_);

    return internal_y;
  }

  DEV_HOST_INLINE typename cuda_vec<internal_coord_t>::type_2d Scale(
      const typename cuda_vec<coord_t>::type_2d& p) const {
    return {ScaleX(p.x), ScaleY(p.y)};
  }

  DEV_HOST_INLINE coord_t UnscaleX(internal_coord_t internal_x) const {
    return internal_x * rrx_ + ddeltax_;
  }

  DEV_HOST_INLINE coord_t UnscaleY(internal_coord_t internal_y) const {
    return internal_y * rry_ + ddeltay_;
  }

  DEV_HOST_INLINE typename cuda_vec<coord_t>::type_2d Unscale(
      const typename cuda_vec<internal_coord_t>::type_2d& p) const {
    return {UnscaleX(p.x), UnscaleY(p.y)};
  }

  DEV_HOST_INLINE internal_coord_t get_internal_min() const {
    return internal_min_;
  }

  DEV_HOST_INLINE internal_coord_t get_internal_max() const {
    return internal_max_;
  }

  DEV_HOST_INLINE internal_coord_t get_internal_range() const {
    return internal_range_;
  }

  DEV_HOST_INLINE bool IsInRange(double v) const {
    return v >= internal_min_ && v <= internal_max_;
  }

 private:
  internal_coord_t internal_max_, internal_min_;
  internal_coord_t internal_range_;
  coord_t rx_, ry_;
  coord_t rrx_, rry_;
  coord_t deltax_, deltay_;
  coord_t ddeltax_, ddeltay_;
};

}  // namespace rayjoin

#endif  // RAYJOIN_MAP_SCALING_H
