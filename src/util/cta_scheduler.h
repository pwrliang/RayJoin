#ifndef RAYJOIN_CTA_SCHEDULER_H
#define RAYJOIN_CTA_SCHEDULER_H
#include <cub/cub.cuh>

#include "util/util.h"

namespace rayjoin {
namespace dev {
typedef uint32_t index_type;

template <const int WARPS_PER_TB, typename TMetaData>
struct warp_np {
  volatile index_type owner[WARPS_PER_TB];
  volatile index_type size[WARPS_PER_TB];
  volatile TMetaData meta_data[WARPS_PER_TB];
};

template <typename TMetaData>
struct tb_np {
  index_type owner;
  index_type size;
  TMetaData meta_data;
};

struct empty_np {};

template <typename ts_type, typename TTB, typename TWP, typename TFG = empty_np>
union np_shared {
  // for scans
  ts_type temp_storage;

  // for tb-level np
  TTB tb;

  // for warp-level np
  TWP warp;

  TFG fg;
};

/*
 * @brief A structure representing a scheduled chunk of work
 */
template <typename TMetaData>
struct np_local {
  index_type size;  // work size
  TMetaData meta_data;
};

// TODO(liang): revisit CTAWorkScheduler and #define NO_CTA_WARP_INTRINSICS
template <typename TMetaData>
struct CTAWorkScheduler {
  template <typename TWork>
  __device__ __forceinline__ static void schedule(np_local<TMetaData>& np_local,
                                                  TWork work) {
    const int WP_SIZE = CUB_PTX_WARP_THREADS;
    const int TB_SIZE = blockDim.x;

    const int NP_WP_CROSSOVER = WP_SIZE;
    const int NP_TB_CROSSOVER = TB_SIZE;

#ifndef NO_CTA_WARP_INTRINSICS
    typedef union np_shared<empty_np, tb_np<TMetaData>, empty_np>
        np_shared_type;
#else
    typedef union np_shared<empty_np, tb_np<TMetaData>, warp_np<32, TMetaData>>
        np_shared_type;   // 32 is max number of warps in block
#endif

    __shared__ np_shared_type np_shared;

    if (threadIdx.x == 0) {
      np_shared.tb.owner = TB_SIZE + 1;
    }

    __syncthreads();

    //
    // First scheduler: processing high-degree work items using the entire block
    //
    while (true) {
      if (np_local.size >= NP_TB_CROSSOVER) {
        // 'Elect' one owner for the entire thread block
        np_shared.tb.owner = threadIdx.x;
      }

      __syncthreads();

      if (np_shared.tb.owner == TB_SIZE + 1) {
        // No owner was elected, i.e. no high-degree work items remain

#ifndef NO_CTA_WARP_INTRINSICS
        // No need to sync threads before moving on to WP scheduler
        // because it does not use shared memory
#else
        __syncthreads();  // Necessary do to the shared memory union used by
                          // both TB and WP schedulers
#endif
        break;
      }

      if (np_shared.tb.owner == threadIdx.x) {
        // This thread is the owner
        np_shared.tb.size = np_local.size;
        np_shared.tb.meta_data = np_local.meta_data;

        // Mark this work-item as processed for future schedulers
        np_local.size = 0;
      }

      __syncthreads();

      auto size = np_shared.tb.size;
      auto meta_data = np_shared.tb.meta_data;

      if (np_shared.tb.owner == threadIdx.x) {
        np_shared.tb.owner = TB_SIZE + 1;
      }

      // Use all threads in thread block to execute individual work
      for (int ii = threadIdx.x; ii < size; ii += TB_SIZE) {
        work(ii, meta_data);
      }

      __syncthreads();
    }

    //
    // Second scheduler: tackle medium-degree work items using the warp
    //
#ifdef NO_CTA_WARP_INTRINSICS
    const int warp_id = threadIdx.x / WP_SIZE;
#endif
    const int lane_id = cub::LaneId();

    while (__any_sync(0xffffffff, np_local.size >= NP_WP_CROSSOVER)) {
#ifndef NO_CTA_WARP_INTRINSICS
      // Compete for work scheduling
      int mask =
          __ballot_sync(0xffffffff, np_local.size >= NP_WP_CROSSOVER ? 1 : 0);
      // Select a deterministic winner
      int leader = __ffs(mask) - 1;

      // Broadcast data from the leader
      auto size = cub::ShuffleIndex<WP_SIZE>(np_local.size, leader, 0xffffffff);
      auto meta_data =
          cub::ShuffleIndex<WP_SIZE>(np_local.meta_data, leader, 0xffffffff);

      if (leader == lane_id) {
        // Mark this work-item as processed
        np_local.size = 0;
      }
#else
      if (np_local.size >= NP_WP_CROSSOVER) {
        // Again, race to select an owner for warp
        np_shared.warp.owner[warp_id] = lane_id;
      }

      cub::WARP_SYNC(0xffffffff);

      if (np_shared.warp.owner[warp_id] == lane_id) {
        // This thread is owner
        np_shared.warp.size[warp_id] = np_local.size;
        np_shared.warp.meta_data[warp_id] = np_local.meta_data;

        // Mark this work-item as processed
        np_local.size = 0;
      }

      cub::WARP_SYNC(0xffffffff);

      auto size = np_shared.warp.size[warp_id];
      auto meta_data = np_shared.warp.meta_data[warp_id];
#endif

      for (int ii = lane_id; ii < size; ii += WP_SIZE) {
        work(ii, meta_data);
      }

      // cub::WARP_SYNC(0xffffffff);
    }

    __syncthreads();

    //
    // Third scheduler: tackle all work-items with size < 32 serially
    //
    // We did not implement the FG (Finegrained) scheduling for simplicity
    // It is possible to disable this scheduler by setting NP_WP_CROSSOVER to 0
    while (__any_sync(0xffffffff, np_local.size > 0)) {
      int mask = __ballot_sync(0xffffffff, np_local.size > 0 ? 1 : 0);
      int leader = __ffs(mask) - 1;

      auto size = cub::ShuffleIndex<WP_SIZE>(np_local.size, leader, mask);
      auto meta_data =
          cub::ShuffleIndex<WP_SIZE>(np_local.meta_data, leader, mask);

      if (leader == lane_id) {
        np_local.size = 0;
      }

      if (lane_id < size) {
        work(lane_id, meta_data);
      }
    }
  }
};
}  // namespace dev
}  // namespace rayjoin
#endif  // RAYJOIN_CTA_SCHEDULER_H
