#ifndef RAYJOIN_MAP_BOUNDING_BOX_H
#define RAYJOIN_MAP_BOUNDING_BOX_H
#include "util/util.h"
namespace rayjoin {
template <typename COORD_T>
struct BoundingBox {
  COORD_T min_x, min_y, max_x, max_y;
  DEV_HOST BoundingBox()
      : min_x(std::numeric_limits<COORD_T>::max()),
        min_y(std::numeric_limits<COORD_T>::max()),
        max_x(-std::numeric_limits<COORD_T>::max()),
        max_y(-std::numeric_limits<COORD_T>::max()) {}
};
}  // namespace rayjoin
#endif  // RAYJOIN_MAP_BOUNDING_BOX_H
