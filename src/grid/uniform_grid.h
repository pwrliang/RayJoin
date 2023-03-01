#ifndef RAYJOIN_GRID_UNIFORM_GRID_H
#define RAYJOIN_GRID_UNIFORM_GRID_H
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "context.h"
#include "grid/cell.h"
#include "util/queue.h"
#include "util/rational.h"
#include "util/stopwatch.h"
#include "util/util.h"
#include "util/vec_math.h"

#define SQ(x) ((x) * (x))
#define SIZEOF_ELEM(vec) \
  (sizeof(typename std::remove_reference<decltype(vec)>::type::value_type))
#define COUNT_CONTAINER_BYTES(vec) (vec.size() * SIZEOF_ELEM(vec))
namespace std {

template <>
struct hash<double2> {
  size_t operator()(const double2& p) const {
    return *reinterpret_cast<const int64_t*>(&p.x) ^
           *reinterpret_cast<const int64_t*>(&p.y);
  }
};
bool operator==(const double2& p1, const double2& p2) {
  return p1.x == p2.x && p1.y == p2.y;
}

}  // namespace std

namespace rayjoin {

namespace dev {

template <typename SCALING_T, typename MAP_T, typename FUNC_T>
DEV_INLINE void iterate_cell(uint32_t grid_size, MAP_T d_map,
                             const SCALING_T& scaling, FUNC_T func) {
  auto internal_min = scaling.get_internal_min();
  auto internal_max = scaling.get_internal_max();
  auto ne = d_map.get_edges_num();

  for (unsigned int eid = TID_1D; eid < ne; eid += TOTAL_THREADS_1D) {
    auto& e = d_map.get_edge(eid);
    auto& p1 = d_map.get_point(e.p1_idx);
    auto& p2 = d_map.get_point(e.p2_idx);
    auto x1 = p1.x, y1 = p1.y;
    auto x2 = p2.x, y2 = p2.y;

    assert(x1 >= internal_min && x1 <= internal_max);
    assert(y1 >= internal_min && y1 <= internal_max);
    assert(x2 >= internal_min && x2 <= internal_max);
    assert(y2 >= internal_min && y2 <= internal_max);

    auto ix1 = calculate_cell(grid_size, scaling, x1),
         iy1 = calculate_cell(grid_size, scaling, y1);
    auto ix2 = calculate_cell(grid_size, scaling, x2),
         iy2 = calculate_cell(grid_size, scaling, y2);

    assert(ix1 >= 0 && ix1 < grid_size);
    assert(iy1 >= 0 && iy1 < grid_size);
    assert(ix2 >= 0 && ix2 < grid_size);
    assert(iy2 >= 0 && iy2 < grid_size);

    // make sure ix1 <= ix2; iy1 <= iy2, so we can iterate a sub grid
    if (ix1 > ix2)
      SWAP(ix1, ix2)
    if (iy1 > iy2)
      SWAP(iy1, iy2)

    for (auto i = ix1; i <= ix2; i++) {
      for (auto j = iy1; j <= iy2; j++) {
        func(i, j, e);
      }
    }
  }
#undef CELL
}

class UniformGrid {
 public:
  using cell_t = dev::Cell;

  UniformGrid() = default;

  DEV_HOST UniformGrid(uint32_t gsize, ArrayView<cell_t> cells,
                       ArrayView<size_t> eids)
      : gsize_(gsize), cells_(cells), eids_(eids) {}

  DEV_INLINE size_t get_eid(size_t offset) const { return eids_[offset]; }

  DEV_INLINE const cell_t& get_cell(size_t cell_idx) const {
    return cells_[cell_idx];
  }

  DEV_INLINE const cell_t& get_cell(uint32_t x, uint32_t y) const {
    assert(x < gsize_);
    assert(y < gsize_);
    return cells_[y * gsize_ + x];
  }

  DEV_HOST_INLINE ArrayView<cell_t> get_cells() const { return cells_; }

  DEV_HOST_INLINE uint32_t get_grid_size() const { return gsize_; }

 private:
  uint32_t gsize_;
  // Input
  ArrayView<cell_t> cells_;
  /* number of edges per cell */
  ArrayView<size_t> eids_;  // eids per cell
};
}  // namespace dev

class UniformGrid {
 public:
  using cell_t = dev::Cell;

  explicit UniformGrid(unsigned int gsize) : gsize_(gsize) {}

  virtual ~UniformGrid() = default;

  template <typename CONTEXT_T>
  void AddMapsToGrid(CONTEXT_T& ctx, bool profiling = false) {
    using edge_t = dev::Edge<typename CONTEXT_T::coefficient_t>;
    Stopwatch sw;
    double t_alloc = 0, t_count_edges = 0, t_compute_offset = 0,
           t_fill_edges = 0;

    auto& stream = ctx.get_stream();
    auto& scaling = ctx.get_scaling();
    auto gsize = gsize_;

    sw.start();
    auto n_cells = get_cells_num();

    cells_.resize(n_cells);
    n_edges_.resize(n_cells);
    row_index_.resize(n_cells + 1, 0);
    VLOG(1) << "Creating the uniform grid, resolution: " << gsize_ << ", Size: "
            << (COUNT_CONTAINER_BYTES(cells_) +
                COUNT_CONTAINER_BYTES(n_edges_) +
                COUNT_CONTAINER_BYTES(row_index_)) /
                   1024 / 1024
            << " MB";
    sw.stop();
    t_alloc = sw.ms();

    ArrayView<cell_t> d_cells(cells_);
    ArrayView<size_t> d_row_index(row_index_);

    sw.start();
    CUDA_CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(cells_.data()), 0,
                               sizeof(cell_t) * cells_.size(),
                               stream.cuda_stream()));

    // Counter number of edges per cell
    FOR2 {
      auto d_map = ctx.get_map(im)->DeviceObject();
      // Count
      LaunchKernel(stream, [=] __device__() mutable {
        dev::iterate_cell(gsize, d_map, scaling,
                          [=](int x, int y, const edge_t& e) mutable {
                            uint32_t cell_idx = y * gsize + x;
                            auto& cell = d_cells[cell_idx];

                            atomicAdd(&cell.ne[im], 1);
                          });
      });
    }
    if (profiling) {
      stream.Sync();
    }
    sw.stop();
    t_count_edges = sw.ms();

    sw.start();
    // Fill ne1 + ne2 to n_edges_
    thrust::transform(
        thrust::cuda::par.on(stream.cuda_stream()), cells_.begin(),
        cells_.end(), n_edges_.begin(),
        [] __device__(const cell_t& cell) { return cell.ne[0] + cell.ne[1]; });

    // Prefix sum
    thrust::inclusive_scan(thrust::cuda::par.on(stream.cuda_stream()),
                           n_edges_.begin(), n_edges_.end(),
                           row_index_.begin() + 1);
    sw.stop();
    stream.Sync();
    t_compute_offset = sw.ms();
    size_t n_edges_cells = row_index_.back();

    sw.start();
    eids_.resize(n_edges_cells);
    sw.stop();
    t_alloc += sw.ms();

    VLOG(1) << "Total Edges in the grid: " << n_edges_cells
            << ", Size: " << COUNT_CONTAINER_BYTES(eids_) / 1024 / 1024
            << " MB";

    ArrayView<size_t> d_eids(eids_);
    sw.start();
    // start filling edges
    FOR2 {
      auto d_map = ctx.get_map(im)->DeviceObject();
      // Count
      LaunchKernel(stream, [=] __device__() mutable {
        dev::iterate_cell(
            gsize, d_map, scaling, [=](int x, int y, const edge_t& e) mutable {
              auto cell_idx = y * gsize + x;
              auto& cell = d_cells[cell_idx];
              auto offset = d_row_index[cell_idx] + atomicAdd(&cell.begin, 1);

              d_eids[offset] = e.eid;
            });
      });
    }

    // restore begin
    LaunchKernel(stream, [=] __device__() mutable {
      for (unsigned int i = TID_1D; i < d_cells.size(); i += TOTAL_THREADS_1D) {
        d_cells[i].begin = d_row_index[i];
      }
    });
    stream.Sync();
    sw.stop();
    t_fill_edges = sw.ms();
    if (profiling) {
      auto total = t_alloc + t_count_edges + t_compute_offset + t_fill_edges;
      printf(
          "Uniform Grid Profiling result:\nAlloc: %.3lf\nCount Edge: "
          "%.3lf\nCompute Offset: %.3lf\nFill Edges: %.3lf\nTotal: %.3lf\n",
          t_alloc, t_count_edges, t_compute_offset, t_fill_edges, total);
    }
  }

  template <typename CONTEXT_T>
  void AddMapToGrid(CONTEXT_T& ctx, int im, bool profiling = false) {
    using edge_t = dev::Edge<typename CONTEXT_T::coefficient_t>;
    Stopwatch sw;
    double t_alloc = 0, t_count_edges = 0, t_compute_offset = 0,
           t_fill_edges = 0;

    auto& stream = ctx.get_stream();
    auto& scaling = ctx.get_scaling();
    auto gsize = gsize_;

    sw.start();
    auto n_cells = get_cells_num();

    cells_.resize(n_cells);
    n_edges_.resize(n_cells);
    row_index_.resize(n_cells + 1);

    thrust::fill_n(row_index_.begin(), row_index_.size(), 0);

    VLOG(1) << "Creating the uniform grid, resolution: " << gsize_ << ", Size: "
            << (COUNT_CONTAINER_BYTES(cells_) +
                COUNT_CONTAINER_BYTES(n_edges_) +
                COUNT_CONTAINER_BYTES(row_index_)) /
                   1024 / 1024
            << " MB";
    sw.stop();
    t_alloc = sw.ms();

    ArrayView<cell_t> d_cells(cells_);
    ArrayView<size_t> d_row_index(row_index_);

    sw.start();
    CUDA_CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(cells_.data()), 0,
                               sizeof(cell_t) * cells_.size(),
                               stream.cuda_stream()));

    // Counter number of edges per cell
    auto d_map = ctx.get_map(im)->DeviceObject();
    // Count
    LaunchKernel(stream, [=] __device__() mutable {
      dev::iterate_cell(gsize, d_map, scaling,
                        [=](int x, int y, const edge_t& e) mutable {
                          uint32_t cell_idx = y * gsize + x;
                          auto& cell = d_cells[cell_idx];

                          atomicAdd(&cell.ne[im], 1);
                        });
    });
    if (profiling) {
      stream.Sync();
    }
    sw.stop();
    t_count_edges = sw.ms();

    sw.start();
    // Fill ne1 + ne2 to n_edges_
    thrust::transform(
        thrust::cuda::par.on(stream.cuda_stream()), cells_.begin(),
        cells_.end(), n_edges_.begin(),
        [=] __device__(const cell_t& cell) { return cell.ne[0] + cell.ne[1]; });

    // Prefix sum
    thrust::inclusive_scan(thrust::cuda::par.on(stream.cuda_stream()),
                           n_edges_.begin(), n_edges_.end(),
                           row_index_.begin() + 1);
    sw.stop();
    stream.Sync();
    t_compute_offset = sw.ms();
    size_t n_edges_cells = row_index_.back();

    sw.start();
    eids_.resize(n_edges_cells);
    sw.stop();
    t_alloc += sw.ms();

    VLOG(1) << "Total Edges in the grid: " << n_edges_cells
            << ", Size: " << COUNT_CONTAINER_BYTES(eids_) / 1024 / 1024
            << " MB";

    ArrayView<size_t> d_eids(eids_);
    sw.start();
    // start filling edges
    // Count
    LaunchKernel(stream, [=] __device__() mutable {
      dev::iterate_cell(
          gsize, d_map, scaling, [=](int x, int y, const edge_t& e) mutable {
            auto cell_idx = y * gsize + x;
            auto& cell = d_cells[cell_idx];
            auto offset = d_row_index[cell_idx] + atomicAdd(&cell.begin, 1);

            d_eids[offset] = e.eid;
          });
    });

    // restore begin
    LaunchKernel(stream, [=] __device__() mutable {
      for (unsigned int i = TID_1D; i < d_cells.size(); i += TOTAL_THREADS_1D) {
        d_cells[i].begin = d_row_index[i];
      }
    });
    stream.Sync();
    sw.stop();
    t_fill_edges = sw.ms();
    if (profiling) {
      auto total = t_alloc + t_count_edges + t_compute_offset + t_fill_edges;
      printf(
          "Uniform Grid Profiling result:\nAlloc: %.3lf\nCount Edge: "
          "%.3lf\nCompute Offset: %.3lf\nFill Edges: %.3lf\nTotal: %.3lf\n",
          t_alloc, t_count_edges, t_compute_offset, t_fill_edges, total);
    }
  }

  void PrintHistogram() const {
    pinned_vector<cell_t> cells(cells_);
    size_t total_edges[3] = {0, 0, 0};  // map0, map1, total
    size_t max_edges_cell = 0;

    for (auto& cell : cells) {
      FOR2 { total_edges[im] += cell.ne[im]; }
      total_edges[2] += cell.ne[0] + cell.ne[1];
      max_edges_cell =
          std::max(max_edges_cell, (size_t) (cell.ne[0] + cell.ne[1]));
    }

    int n_group = 10;
    size_t group_size = (max_edges_cell + n_group - 1) / n_group;
    std::vector<size_t> n_cells_in_group[3], n_edges_in_group[3];

    auto to_group_id = [&](size_t val) {
      for (int ig = 0; ig < n_group; ig++) {
        auto begin = group_size * ig;
        auto end = group_size * (ig + 1);

        if (val >= begin && val < end) {
          return ig;
        }
      }
      return n_group;
    };

    for (int im = 0; im < 3; im++) {
      n_cells_in_group[im].resize(n_group, 0);
      n_edges_in_group[im].resize(n_group, 0);

      for (auto& cell : cells) {
        auto ne = im < 2 ? cell.ne[im] : cell.ne[0] + cell.ne[1];
        auto gid = to_group_id(ne);

        n_cells_in_group[im][gid]++;
        n_edges_in_group[im][gid] += ne;
      }
    }
    size_t total_n_cells = gsize_ * gsize_;
    size_t empty_n_cells = 0;

    for (int gid = 0; gid < n_group; gid++) {
      auto n_cells1 = n_cells_in_group[0][gid];
      auto n_cells2 = n_cells_in_group[1][gid];
      auto n_cells = n_cells_in_group[2][gid];
      auto begin = group_size * gid;
      auto end = group_size * (gid + 1);

      if (n_cells == 0) {
        empty_n_cells++;
      }
      //      printf(
      //          "group: %d, [%lu, %lu), n_cells1: %lu, ratio: %.2f, n_cells2:
      //          %lu, " "ratio: %.2f, n_cells: %lu, ratio: %.2f\n", gid, begin,
      //          end, n_cells1, (float) n_cells1 / total_n_cells, n_cells2,
      //          (float) n_cells2 / total_n_cells, n_cells,
      //          (float) n_cells / total_n_cells);
      printf(
          "group: %d, [%lu, %lu), n_cells: %lu, ratio: %2.2f%% contri: "
          "%2.2f%%\n",
          gid, begin, end, n_cells, (float) n_cells / total_n_cells * 100,
          (float) n_edges_in_group[2][gid] / total_edges[2] * 100);
    }
    printf("Total cells: %lu, Empty cells: %lu\n", total_n_cells,
           empty_n_cells);
  }

  void DumpCells(const std::string& out_prefix) const {
    pinned_vector<cell_t> cells(cells_);

    if (!out_prefix.empty()) {
      FOR2 {
        std::ofstream of(out_prefix + "_map" + std::to_string(im) +
                         "_cells.txt");

        for (int i = 0; i < gsize_; i++) {
          for (int j = 0; j < gsize_; j++) {
            auto idx = i * gsize_ + j;
            auto ne = cells[idx].ne[im];
            of << ne << " ";
          }
          of << std::endl;
        }
        of.close();
      }
    }
  }

  size_t get_grid_size() const { return gsize_; }

  size_t get_cells_num() const { return gsize_ * gsize_; }

  dev::UniformGrid DeviceObject() const {
    return dev::UniformGrid(gsize_, ArrayView<cell_t>(cells_),
                            ArrayView<size_t>(eids_));
  }

  ArrayView<size_t> get_eids() { return eids_; }

  void KernelSizingGrid(dim3& grid_dim, dim3& block_dim) const {
    unsigned int cu_block_size = MAX_BLOCK_SIZE_2D;
    unsigned int cu_grid_size = (gsize_ + cu_block_size - 1) / cu_block_size;

    grid_dim = {cu_grid_size, cu_grid_size, 1};
    block_dim = {cu_block_size, cu_block_size, 1};
  }

 protected:
  uint32_t gsize_;
  // Input
  thrust::device_vector<cell_t> cells_;
  /* number of edges per cell */
  thrust::device_vector<size_t> eids_;  // eids per cell
  // tmp
  thrust::device_vector<size_t> row_index_;
  thrust::device_vector<size_t> n_edges_;  // n edges per cell
};
}  // namespace rayjoin

#endif  // RAYJOIN_GRID_UNIFORM_GRID_H
