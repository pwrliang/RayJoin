#ifndef RAYJOIN_CONFIG_H
#define RAYJOIN_CONFIG_H
#define DONTKNOW (-1)
#define SCALING_BOUNDING_BOX_MARGIN (1)
#define RAY_TMAX (1e10)
#define PRIMITIVE_HEIGHT (0.01)
#define EXTERIOR_FACE_ID (0)
#define FOR2 for (int im = 0; im < 2; im++)
namespace rayjoin {
using index_t = uint32_t;
using polygon_id_t = int;
using coord_t = double;
using coefficient_t = __int128;
}  // namespace rayjoin

#endif  // RAYJOIN_CONFIG_H
