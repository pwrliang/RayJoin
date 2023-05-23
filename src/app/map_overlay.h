
#ifndef RAYJOIN_APP_MAP_OVERLAY_H
#define RAYJOIN_APP_MAP_OVERLAY_H
#include "algo/lsi.h"
#include "app/lsi.h"
#include "app/pip.h"

namespace rayjoin {
template <typename CONTEXT_T>
class MapOverlay {
 public:
  using coord_t = typename CONTEXT_T::coord_t;
  using internal_coord_t = typename CONTEXT_T::internal_coord_t;
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using xsect_t = dev::Intersection<internal_coord_t>;

  explicit MapOverlay(CONTEXT_T& ctx) : ctx_(ctx) {}

  virtual void Init() = 0;

  virtual void BuildIndex() = 0;

  virtual void IntersectEdge(int query_map_id) = 0;

  virtual void LocateVerticesInOtherMap(int query_map_id) = 0;

  virtual void ComputeOutputPolygons() = 0;

  virtual void WriteResult(const char* path) = 0;

  thrust::host_vector<xsect_t> get_xsect_edges() const {
    thrust::host_vector<xsect_t> res;

    lsi_->CopyTo(res);
    return res;
  }

  thrust::host_vector<index_t> get_closet_eids(int im) const {
    thrust::host_vector<index_t> res = closest_eids_[im];

    return res;
  }

  thrust::host_vector<index_t> get_point_in_polygon(int im) const {
    thrust::host_vector<index_t> res = point_in_polygon_[im];

    return res;
  }

 protected:
  CONTEXT_T& ctx_;
  std::shared_ptr<LSI<CONTEXT_T>> lsi_;
  std::shared_ptr<PIP<CONTEXT_T>> pip_;
  thrust::device_vector<polygon_id_t> closest_eids_[2];
  thrust::device_vector<polygon_id_t> point_in_polygon_[2];
};

}  // namespace rayjoin

#endif  // RAYJOIN_APP_MAP_OVERLAY_H
