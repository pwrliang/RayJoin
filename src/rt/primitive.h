#ifndef RAYJOIN_RT_PRIMITIVE_H
#define RAYJOIN_RT_PRIMITIVE_H
#include <thrust/device_vector.h>

#include <cub/cub.cuh>  // or equivalently <cub/block/block_radix_sort.cuh>

#include "config.h"
#include "util/array_view.h"
#include "util/helpers.h"
#include "util/queue.h"
#include "util/shared_value.h"
#include "util/stream.h"
#include "util/util.h"

namespace rayjoin {
namespace dev {
template <typename MAP_T, typename SCALING_T>
DEV_HOST_INLINE OptixAabb GetOptixAabb(const MAP_T& d_map,
                                       const SCALING_T& scaling, size_t eid) {
  const auto& e = d_map.get_edge(eid);
  auto p1_idx = e.p1_idx;
  auto p2_idx = e.p2_idx;
  const auto& p1 = d_map.get_point(p1_idx);
  const auto& p2 = d_map.get_point(p2_idx);
  auto x1 = scaling.UnscaleX(p1.x);
  auto y1 = scaling.UnscaleY(p1.y);
  auto x2 = scaling.UnscaleX(p2.x);
  auto y2 = scaling.UnscaleY(p2.y);

  OptixAabb aabb;

  aabb.minX = std::min(x1, x2);
  aabb.maxX = std::max(x1, x2);
  aabb.minY = std::min(y1, y2);
  aabb.maxY = std::max(y1, y2);
  aabb.minZ = -PRIMITIVE_HEIGHT / 2;
  aabb.maxZ = PRIMITIVE_HEIGHT / 2;
  return aabb;
}

DEV_HOST_INLINE OptixAabb OptixAabbMerge(const OptixAabb& box1,
                                         const OptixAabb& box2) {
  OptixAabb aabb;

  aabb.minX = std::min(box1.minX, box2.minX);
  aabb.maxX = std::max(box1.maxX, box2.maxX);
  aabb.minY = std::min(box1.minY, box2.minY);
  aabb.maxY = std::max(box1.maxY, box2.maxY);
  aabb.minZ = std::min(box1.minZ, box2.minZ);
  aabb.maxZ = std::max(box1.maxZ, box2.maxZ);
  return aabb;
}

DEV_HOST_INLINE float OptixAabbArea(const OptixAabb& box) {
  return (box.maxX - box.minX) * (box.maxY - box.minY);
}

DEV_HOST_INLINE bool OptixAabbIsWithin(const OptixAabb& base,
                                       const OptixAabb& to_test) {
  return to_test.minX >= base.minX && to_test.maxX <= base.maxX &&
         to_test.minY >= base.minY && to_test.maxY <= base.maxY &&
         to_test.minZ >= base.minZ && to_test.maxZ <= base.maxZ;
}

DEV_HOST_INLINE OptixAabb OptixAabbRoundingUp(const OptixAabb& aabb) {
  OptixAabb rounded_aabb;

  rounded_aabb.minX = next_float_from_double(aabb.minX, -1, ROUNDING_ITER);
  rounded_aabb.maxX = next_float_from_double(aabb.maxX, 1, ROUNDING_ITER);
  rounded_aabb.minY = next_float_from_double(aabb.minY, -1, ROUNDING_ITER);
  rounded_aabb.maxY = next_float_from_double(aabb.maxY, 1, ROUNDING_ITER);
  rounded_aabb.minZ = next_float_from_double(aabb.minZ, -1, ROUNDING_ITER);
  rounded_aabb.maxZ = next_float_from_double(aabb.maxZ, 1, ROUNDING_ITER);
  return rounded_aabb;
}

struct Uint2Less {
  template <typename uint2>
  __device__ bool operator()(const uint2& lhs, const uint2& rhs) {
    return lhs.x < rhs.x;
  }
};

}  // namespace dev

template <typename MAP_T, typename SCALING_T>
void FillPrimitives(Stream& stream, const MAP_T& d_map,
                    const SCALING_T& scaling,
                    thrust::device_vector<OptixAabb>& aabbs) {
  auto ne = d_map.get_edges_num();
  aabbs.resize(ne);
  ArrayView<OptixAabb> d_aabbs(aabbs);

  ForEach(stream, ne, [=] __device__(size_t eid) mutable {
    const auto& e = d_map.get_edge(eid);
    auto p1_idx = e.p1_idx;
    auto p2_idx = e.p2_idx;
    const auto& p1 = d_map.get_point(p1_idx);
    const auto& p2 = d_map.get_point(p2_idx);
    auto x1 = scaling.UnscaleX(p1.x);
    auto y1 = scaling.UnscaleY(p1.y);
    auto x2 = scaling.UnscaleX(p2.x);
    auto y2 = scaling.UnscaleY(p2.y);
    auto& aabb = d_aabbs[eid];

    aabb.minX = next_float_from_double(min(x1, x2), -1, ROUNDING_ITER);
    aabb.maxX = next_float_from_double(max(x1, x2), 1, ROUNDING_ITER);
    aabb.minY = next_float_from_double(min(y1, y2), -1, ROUNDING_ITER);
    aabb.maxY = next_float_from_double(max(y1, y2), 1, ROUNDING_ITER);
    aabb.minZ = -PRIMITIVE_HEIGHT / 2;
    aabb.maxZ = PRIMITIVE_HEIGHT / 2;
  });
}

template <typename MAP_T, typename SCALING_T>
void FillPrimitivesGroupNew(
    Stream& stream, const MAP_T& d_map, const SCALING_T& scaling,
    uint32_t max_iter, float area_enlarge,
    thrust::device_vector<OptixAabb>& aabbs,
    thrust::device_vector<thrust::pair<size_t, size_t>>& eid_range) {
  Stopwatch sw;
  sw.start();
  uint32_t ne = d_map.get_edges_num();
  const int block_size = 64;
  dim3 block_dims{block_size, 1, 1};
  dim3 grid_dims{(ne + block_dims.x - 1) / block_dims.x, 1, 1};
  SharedValue<uint32_t> g_tail;

  aabbs.resize(ne);
  eid_range.resize(ne);

  ArrayView<OptixAabb> d_aabbs(aabbs);
  ArrayView<thrust::pair<size_t, size_t>> d_eid_range(eid_range);

  g_tail.set(0, stream);
  auto* p_g_tail = g_tail.data();

  LaunchKernel(stream, grid_dims, block_dims, [=] __device__() mutable {
    uint32_t blk_id = blockIdx.x;
    uint32_t eid_begin = std::min(blk_id * blockDim.x, ne);
    uint32_t eid_end = std::min((blk_id + 1) * blockDim.x, ne);
    typedef cub::BlockMergeSort<uint2, block_size, 1, OptixAabb> BlockMergeSort;
    __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;

    __shared__ uint2 range_in[block_size], range_out[block_size];
    __shared__ OptixAabb aabb_in[block_size], aabb_out[block_size];
    __shared__ uint32_t tail_in, tail_out;
    uint2 *p_range_in = range_in, *p_range_out = range_out;
    OptixAabb *p_aabb_in = aabb_in, *p_aabb_out = aabb_out;
    uint32_t *p_tail_in = &tail_in, *p_tail_out = &tail_out;
    uint32_t eid = eid_begin + threadIdx.x;

    if (threadIdx.x == 0) {
      tail_in = eid_end - eid_begin;
      tail_out = 0;
    }
    __syncthreads();

    if (eid < eid_end) {
      range_in[threadIdx.x] = uint2{eid, eid + 1};
      aabb_in[threadIdx.x] = dev::GetOptixAabb(d_map, scaling, eid);
    }

    for (int iter = 0; iter < max_iter; iter++) {
      __syncthreads();
      auto n_in = *p_tail_in;

      if (threadIdx.x < (n_in + 1) / 2) {
        auto idx1 = std::min(threadIdx.x * 2, n_in - 1);
        auto idx2 = std::min(idx1 + 1, n_in - 1);
        auto& range1 = p_range_in[idx1];
        auto& aabb1 = p_aabb_in[idx1];
        auto& range2 = p_range_in[idx2];
        auto& aabb2 = p_aabb_in[idx2];

        if (idx1 == idx2) {  // only 1 range
          auto tail = atomicAdd(p_tail_out, 1);
          assert(tail < block_size);

          p_range_out[tail] = range1;
          p_aabb_out[tail] = aabb1;
        } else {
          OptixAabb merged_aabb = dev::OptixAabbMerge(aabb1, aabb2);
          auto area = max(dev::OptixAabbArea(aabb1), dev::OptixAabbArea(aabb2));
          auto merged_area = dev::OptixAabbArea(merged_aabb);
          assert(range1.y == range2.x);
          if (merged_area / area < area_enlarge) {
            uint2 new_range{range1.x, range2.y};
            auto tail = atomicAdd(p_tail_out, 1);

            assert(tail < block_size);
            p_range_out[tail] = new_range;
            p_aabb_out[tail] = merged_aabb;
          } else {
            auto tail = atomicAdd(p_tail_out, 2);

            assert(tail + 2 <= block_size);
            p_range_out[tail] = range1;
            p_aabb_out[tail] = aabb1;
            p_range_out[tail + 1] = range2;
            p_aabb_out[tail + 1] = aabb2;
          }
        }
      }

      __syncthreads();
      int n_out = *p_tail_out;
      assert(n_out <= block_size);
      thrust::swap(p_range_in, p_range_out);
      thrust::swap(p_aabb_in, p_aabb_out);
      thrust::swap(p_tail_in, p_tail_out);

      if (n_in == n_out) {
        break;
      }

      BlockMergeSort(temp_storage_shuffle)
          .Sort(*static_cast<uint2(*)[1]>(
                    static_cast<void*>(p_range_in + threadIdx.x)),
                *static_cast<OptixAabb(*)[1]>(
                    static_cast<void*>(p_aabb_in + threadIdx.x)),
                dev::Uint2Less(), n_out,
                uint2{std::numeric_limits<uint32_t>::max(),
                      std::numeric_limits<uint32_t>::max()});
      __syncthreads();
      if (threadIdx.x == 0) {
        *p_tail_out = 0;
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) {
      assert(*p_tail_in <= block_size);
      *p_tail_out = atomicAdd(p_g_tail, *p_tail_in);
    }
    __syncthreads();

    if (threadIdx.x < *p_tail_in) {
      auto i = *p_tail_out + threadIdx.x;
      assert(i < ne);
      d_aabbs[i] = dev::OptixAabbRoundingUp(p_aabb_in[threadIdx.x]);
      d_eid_range[i] = thrust::make_pair(p_range_in[threadIdx.x].x,
                                         p_range_in[threadIdx.x].y);
    }
  });
  stream.Sync();
  auto n_aabbs = g_tail.get(stream);
  aabbs.resize(n_aabbs);
  eid_range.resize(n_aabbs);
  sw.stop();

  LOG(INFO) << "Enlarge limit: " << area_enlarge << " ne: " << ne
            << " aabbs: " << n_aabbs
            << " Compress ratio: " << (float) (ne - n_aabbs) / ne
            << " Time: " << sw.ms() << " ms";
}

template <typename MAP_T, typename SCALING_T>
void FillPrimitivesGroup(
    Stream& stream, const MAP_T& d_map, const SCALING_T& scaling,
    uint32_t win_size, float area_enlarge,
    thrust::device_vector<OptixAabb>& aabbs,
    thrust::device_vector<thrust::pair<size_t, size_t>>& eid_range) {
  Stopwatch sw;
  sw.start();
  auto ne = d_map.get_edges_num();
  auto n_wins = (ne + win_size - 1) / win_size;
  thrust::device_vector<uint32_t> n_grp(n_wins);  // win id-># of groups
  thrust::device_vector<size_t> psum_n_grp(n_wins + 1, 0);

  // TODO, each thread merge edges with ids i, i+1
  ForEach(
      stream, n_wins,
      [=] __device__(size_t win_id, ArrayView<uint32_t> d_n_grp) mutable {
        size_t eid_begin = std::min(win_id * win_size, ne);
        size_t eid_end = std::min((win_id + 1) * win_size, ne);
        OptixAabb last_aabb;
        size_t n_groups = 0;

        for (size_t eid = eid_begin; eid < eid_end; eid++) {
          OptixAabb curr_aabb = dev::GetOptixAabb(d_map, scaling, eid);

          if (eid == eid_begin) {
            last_aabb = curr_aabb;
            n_groups++;
          } else {
            OptixAabb merged_aabb = dev::OptixAabbMerge(last_aabb, curr_aabb);
            auto area = dev::OptixAabbArea(last_aabb);
            auto merged_area = dev::OptixAabbArea(merged_aabb);

            if (merged_area / area < area_enlarge) {
              last_aabb = merged_aabb;
            } else {  // start a new group
              last_aabb = curr_aabb;
              n_groups++;
            }
          }
        }

        d_n_grp[win_id] = n_groups;
      },
      ArrayView<uint32_t>(n_grp));

  thrust::inclusive_scan(thrust::cuda::par.on(stream.cuda_stream()),
                         n_grp.begin(), n_grp.end(), psum_n_grp.begin() + 1);
  stream.Sync();
  size_t n_grps = psum_n_grp[psum_n_grp.size() - 1];

  aabbs.resize(n_grps);
  eid_range.resize(n_grps);

  ArrayView<OptixAabb> d_aabbs(aabbs);
  ArrayView<thrust::pair<size_t, size_t>> d_eid_range(eid_range);

  ForEach(
      stream, n_wins,
      [=] __device__(size_t win_id, ArrayView<size_t> d_psum_n_grp) mutable {
        size_t eid_begin = std::min(win_id * win_size, ne);
        size_t eid_end = std::min((win_id + 1) * win_size, ne);
        size_t gid_begin = d_psum_n_grp[win_id];

        OptixAabb last_aabb;
        size_t last_eid;
        size_t n_groups = 0;

        for (size_t eid = eid_begin; eid < eid_end; eid++) {
          OptixAabb curr_aabb = dev::GetOptixAabb(d_map, scaling, eid);

          if (eid == eid_begin) {
            last_aabb = curr_aabb;
            last_eid = eid;
          } else {
            OptixAabb merged_aabb = dev::OptixAabbMerge(last_aabb, curr_aabb);
            auto area = dev::OptixAabbArea(last_aabb);
            auto merged_area = dev::OptixAabbArea(merged_aabb);

            if (merged_area / area < area_enlarge) {
              last_aabb = merged_aabb;
            } else {
              // write back last aabb
              d_aabbs[gid_begin + n_groups] =
                  dev::OptixAabbRoundingUp(last_aabb);
              d_eid_range[gid_begin + n_groups] =
                  thrust::make_pair(last_eid, eid);

#ifndef NDEBUG
              for (auto sub_eid = last_eid; sub_eid < eid; sub_eid++) {
                auto aabb = dev::GetOptixAabb(d_map, scaling, sub_eid);

                assert(dev::OptixAabbIsWithin(d_aabbs[gid_begin + n_groups],
                                              aabb));
              }
#endif
              // new group
              last_aabb = curr_aabb;
              last_eid = eid;
              n_groups++;
            }
          }
        }
        d_aabbs[gid_begin + n_groups] = dev::OptixAabbRoundingUp(last_aabb);
        d_eid_range[gid_begin + n_groups] =
            thrust::make_pair(last_eid, eid_end);

#ifndef NDEBUG
        for (auto sub_eid = last_eid; sub_eid < eid_end; sub_eid++) {
          auto aabb = dev::GetOptixAabb(d_map, scaling, sub_eid);

          assert(dev::OptixAabbIsWithin(d_aabbs[gid_begin + n_groups], aabb));
        }
#endif
      },
      ArrayView<size_t>(psum_n_grp));
  sw.stop();
  LOG(INFO) << "Window size: " << win_size << " Window count: " << n_wins
            << " Enlarge limit: " << area_enlarge << " ne: " << ne
            << " Group count: " << n_grps
            << " Compress ratio: " << (float) (ne - n_grps) / ne
            << " Time: " << sw.ms() << " ms";
}

}  // namespace rayjoin

#endif  // RAYJOIN_RT_PRIMITIVE_H
