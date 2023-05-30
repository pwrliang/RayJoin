#ifndef RT_LAUNCH_PARAMETERS_H
#define RT_LAUNCH_PARAMETERS_H
#include <thrust/tuple.h>

#include <cstdint>

#include "algo/lsi.h"
#include "config.h"
#include "map/map.h"
#include "map/scaling.h"
#include "util/array_view.h"
#include "util/bitset.h"
#include "util/queue.h"

namespace rayjoin {

struct LaunchParamsLSI {
  using coord_t = coord_t;
  using internal_coord_t = detail::internal_coord<coord_t>::type;
  using map_t = dev::Map<internal_coord_t, coefficient_t>;
  using edge_t = typename map_t::edge_t;
  using point_t = typename map_t::point_t;
  using xsect_t = dev::Intersection<internal_coord_t>;

  edge_t* base_edges;
  point_t* base_points;
  thrust::pair<size_t, size_t>* eid_range;

  int query_map_id;
  ArrayView<edge_t> query_edges;  // ray gen map
  point_t* query_points;

  Scaling<coord_t, internal_coord_t, detail::internal_coord<coord_t>::shift>
      scaling;
  // Used for BVH Traversable
  OptixTraversableHandle traversable;

  // Output
  dev::Queue<xsect_t, uint32_t> xsects;
#ifndef NDEBUG
  uint32_t *n_tests;
#endif
};

struct LaunchParamsPIP {
  using coord_t = coord_t;
  using internal_coord_t = detail::internal_coord<coord_t>::type;
  using map_t = dev::Map<internal_coord_t, coefficient_t>;
  using edge_t = typename map_t::edge_t;
  using point_t = typename map_t::point_t;

  edge_t* base_map_edges;
  point_t* base_map_points;
  thrust::pair<size_t, size_t>* eid_range;

  int query_map_id;
  ArrayView<point_t> query_points;
  Scaling<coord_t, internal_coord_t, detail::internal_coord<coord_t>::shift>
      scaling;
  // Used for BVH Traversable
  OptixTraversableHandle traversable;

  // Output
  index_t* closest_eids;

#ifndef NDEBUG
  uint32_t* hit_count;
  uint32_t* closer_count;
  uint32_t* last_update_count;
  uint32_t* fail_update_count;
#endif
};
}  // namespace rayjoin
#endif  // RT_LAUNCH_PARAMETERS_H
