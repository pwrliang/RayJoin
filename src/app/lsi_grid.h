#ifndef APP_LSI_GRID_H
#define APP_LSI_GRID_H
#include <utility>

#include "algo/lsi.h"
#include "app/lsi.h"
#include "config.h"
#include "grid/cell.h"
#include "grid/uniform_grid.h"
#include "map/map.h"
#include "map/scaling.h"
#include "query_config.h"
#include "util/cta_scheduler.h"
#include "util/queue.h"

namespace rayjoin {
namespace dev {

template <typename COORD_T, typename INTERNAL_COORD_T, typename COEFFICIENT_T>
DEV_INLINE void intersect_one_cell(
    int cell_x, int cell_y, dev::UniformGrid& grid,
    const Scaling<COORD_T, INTERNAL_COORD_T>& scaling,
    dev::Map<INTERNAL_COORD_T, COEFFICIENT_T> map1,
    dev::Map<INTERNAL_COORD_T, COEFFICIENT_T> map2,
    dev::Queue<Intersection<INTERNAL_COORD_T>, uint32_t> xsect_edges) {
  auto gsize = grid.get_grid_size();
  const auto& cell = grid.get_cell(cell_x, cell_y);
  auto ne1 = cell.ne[0], ne2 = cell.ne[1];
  auto begin = cell.begin;

  for (auto ie1 = 0; ie1 < ne1; ie1++) {
    auto eid1 = grid.get_eid(begin + ie1);
    const auto& e1 = map1.get_edge(eid1);
    const auto& e1_p1 = map1.get_point(e1.p1_idx);
    const auto& e1_p2 = map1.get_point(e1.p2_idx);

    for (auto ie2 = 0; ie2 < ne2; ie2++) {
      auto eid2 = grid.get_eid(begin + ne1 + ie2);
      const auto& e2 = map2.get_edge(eid2);
      const auto& e2_p1 = map2.get_point(e2.p1_idx);
      const auto& e2_p2 = map2.get_point(e2.p2_idx);

#if 0
#ifndef NDEBUG
      {
        // cast to double to prevent overflow
        auto denom = (double) e1.a * e2.b - (double) e2.a * e1.b;
        auto numx = (double) e2.c * e1.b - (double) e1.c * e2.b;
        auto numy = (double) e2.a * e1.c - (double) e1.a * e2.c;
        static const __uint128_t UINT128_MAX =__uint128_t(__int128_t(-1L));
        static const __int128_t INT128_MAX = UINT128_MAX >> 1;
        static const __int128_t INT128_MIN = -INT128_MAX - 1;


        assert(denom >= INT128_MIN && denom <= INT128_MAX);
        assert(numx >= INT128_MIN && numx <= INT128_MAX);
        assert(numy >= INT128_MIN && numy <= INT128_MAX);
      }
#endif
#endif
      tcb::rational<__int128> xsect_x, xsect_y;
      if (intersect_test(e1, e1_p1, e1_p2, e2, e2_p1, e2_p2, xsect_x,
                         xsect_y)) {
        auto xsect_cell_x = dev::calculate_cell(gsize, scaling, xsect_x);
        auto xsect_cell_y = dev::calculate_cell(gsize, scaling, xsect_y);

        if (xsect_cell_x == cell_x && xsect_cell_y == cell_y) {
          Intersection<INTERNAL_COORD_T> xsect;
          xsect.x = xsect_x;
          xsect.y = xsect_y;
          xsect.eid[0] = eid1;
          xsect.eid[1] = eid2;
          xsect_edges.AppendWarp(xsect);
        }
      }
    }
  }
}

template <typename COORD_T, typename INTERNAL_COORD_T, typename COEFFICIENT_T>
DEV_INLINE void intersect_one_cell_lb(
    int cell_idx, dev::UniformGrid& grid,
    const Scaling<COORD_T, INTERNAL_COORD_T>& scaling,
    dev::Map<INTERNAL_COORD_T, COEFFICIENT_T> map1,
    dev::Map<INTERNAL_COORD_T, COEFFICIENT_T> map2,
    dev::Queue<Intersection<INTERNAL_COORD_T>, uint32_t> xsect_edges) {
  auto gsize = grid.get_grid_size();
  auto work_size = gsize * gsize;
  dev::np_local<index_t> np_local = {0, 0};

  if (cell_idx < work_size) {
    auto& cell = grid.get_cell(cell_idx);
    auto ne1 = cell.ne[0], ne2 = cell.ne[1];

    np_local.size = ne1 * ne2;
    np_local.meta_data = cell_idx;
  }

  dev::CTAWorkScheduler<index_t>::template schedule(
      np_local, [=](index_t edge_2d, index_t cell_idx) mutable {
        const auto& cell = grid.get_cell(cell_idx);
        auto begin = cell.begin;
        auto ne1 = cell.ne[0], ne2 = cell.ne[1];
        auto ie1 = edge_2d / ne2;
        auto ie2 = edge_2d % ne2;
        auto eid1 = grid.get_eid(begin + ie1);
        auto eid2 = grid.get_eid(begin + ne1 + ie2);

        const auto& e1 = map1.get_edge(eid1);
        const auto& e1_p1 = map1.get_point(e1.p1_idx);
        const auto& e1_p2 = map1.get_point(e1.p2_idx);
        const auto& e2 = map2.get_edge(eid2);
        const auto& e2_p1 = map2.get_point(e2.p1_idx);
        const auto& e2_p2 = map2.get_point(e2.p2_idx);
        tcb::rational<__int128> xsect_x, xsect_y;

        if (intersect_test(e1, e1_p1, e1_p2, e2, e2_p1, e2_p2, xsect_x,
                           xsect_y)) {
          auto xsect_cell_x = dev::calculate_cell(gsize, scaling, xsect_x);
          auto xsect_cell_y = dev::calculate_cell(gsize, scaling, xsect_y);

          auto cell_x = cell_idx % gsize;
          auto cell_y = cell_idx / gsize;

          if (xsect_cell_x == cell_x && xsect_cell_y == cell_y) {
            dev::Intersection<INTERNAL_COORD_T> xsect;
            xsect.x = xsect_x;
            xsect.y = xsect_y;
            xsect.eid[0] = eid1;
            xsect.eid[1] = eid2;
            xsect_edges.Append(xsect);
          }
        }
      });
}
}  // namespace dev

template <typename CONTEXT_T>
class LSIGrid : public LSI<CONTEXT_T> {
  using coefficient_t = typename CONTEXT_T::coefficient_t;
  using coord_t = typename LSI<CONTEXT_T>::coord_t;
  using internal_coord_t = typename LSI<CONTEXT_T>::internal_coord_t;
  using xsect_t = typename LSI<CONTEXT_T>::xsect_t;
  using grid_t = UniformGrid;  // only contains 1 map
  using cell_t = typename grid_t::cell_t;
  using map_t = typename CONTEXT_T::map_t;
  using edge_t = typename map_t::edge_t;

 public:
  LSIGrid(CONTEXT_T& ctx, std::shared_ptr<grid_t> grid)
      : LSI<CONTEXT_T>(ctx), grid_(std::move(grid)) {}

  // fixme: respect query map id
  void Query(Stream& stream, int query_map_id) override {
    auto& ctx = this->get_context();
    auto& scaling = ctx.get_scaling();
    auto gsize = grid_->get_grid_size();
    auto d_xsect_queue = this->xsect_queue_.DeviceObject();
    auto d_grid = grid_->DeviceObject();
    auto d_base_map = ctx.get_map(0)->DeviceObject();
    auto d_query_map = ctx.get_map(1)->DeviceObject();
    dim3 grid_dim, block_dim;

    this->xsect_queue_.Clear(stream);

    if (config_.lb) {
      size_t work_size = gsize * gsize;

      KernelSizing(grid_dim, block_dim, work_size);
      LaunchKernel(stream, grid_dim, block_dim, [=] __device__() mutable {
        uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;

        for (uint32_t i = TID_1D; i < work_size_rup; i += TOTAL_THREADS_1D) {
          intersect_one_cell_lb(i, d_grid, scaling, d_base_map, d_query_map,
                                d_xsect_queue);
        }
      });
    } else {
      grid_->KernelSizingGrid(grid_dim, block_dim);

      // Compute intersections
      LaunchKernel(stream, grid_dim, block_dim, [=] __device__() mutable {
        auto cell_x = blockIdx.x * blockDim.x + threadIdx.x;
        auto cell_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (cell_x < gsize && cell_y < gsize) {
          dev::intersect_one_cell<coord_t, internal_coord_t>(
              cell_x, cell_y, d_grid, scaling, d_base_map, d_query_map,
              d_xsect_queue);
        }
      });
    }
#ifndef NDEBUG
    if (config_.profile) {
      SharedValue<uint64_t> total_ne1, total_ne2;
      auto *d_total_ne1 = total_ne1.data(), *d_total_ne2 = total_ne2.data();
      auto* d_pro_counter = this->prof_counter_.data();

      total_ne1.set(0, stream);
      total_ne2.set(0, stream);
      this->prof_counter_.set(0, stream);
      grid_->KernelSizingGrid(grid_dim, block_dim);

      LaunchKernel(stream, grid_dim, block_dim, [=] __device__() mutable {
        auto cell_x = blockIdx.x * blockDim.x + threadIdx.x;
        auto cell_y = blockIdx.y * blockDim.y + threadIdx.y;

        if (cell_x < gsize && cell_y < gsize) {
          const auto& cell = d_grid.get_cell(cell_x, cell_y);
          auto ne1 = cell.ne[0], ne2 = cell.ne[1];

          atomicAdd(reinterpret_cast<unsigned long long int*>(d_total_ne1),
                    ne1);
          atomicAdd(reinterpret_cast<unsigned long long int*>(d_total_ne2),
                    ne2);
          atomicAdd(reinterpret_cast<unsigned long long int*>(d_pro_counter),
                    ne1 * ne2);
        }
      });
      LOG(INFO) << "Total ne1: " << total_ne1.get(stream);
      LOG(INFO) << "Total ne2: " << total_ne2.get(stream);
      LOG(INFO) << "Total tests: " << this->prof_counter_.get(stream);
      LOG(INFO) << "Ave #E per cell: "
                << (total_ne1.get(stream) + total_ne2.get(stream)) /
                       (gsize * gsize);
    }
#endif
    stream.Sync();
//    grid_->PrintHistogram1();
  }

  void set_config(const QueryConfigGrid& config) { config_ = config; }

  std::shared_ptr<grid_t> get_grid() { return grid_; }

 private:
  std::shared_ptr<grid_t> grid_;
  QueryConfigGrid config_;
};
}  // namespace rayjoin
#endif  // APP_LSI_GRID_H
