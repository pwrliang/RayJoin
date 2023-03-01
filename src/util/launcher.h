#ifndef RAYJOIN_LAUNCHER_H
#define RAYJOIN_LAUNCHER_H
#include "util/stream.h"
#include "util/util.h"
namespace rayjoin {
template <typename F, typename... Args>
__global__ void KernelWrapper(F f, Args... args) {
  f(args...);
}

template <typename F, typename... Args>
__global__ void KernelWrapperForEach(size_t size, F f, Args... args) {
  for (size_t i = TID_1D; i < size; i += TOTAL_THREADS_1D) {
    f(i, args...);
  }
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, dim3 grid_size, dim3 block_size, F f,
                  Args&&... args) {
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, F f, Args&&... args) {
  int grid_size, block_size;

  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
      &grid_size, &block_size, KernelWrapper<F, Args...>, 0,
      reinterpret_cast<int>(MAX_BLOCK_SIZE)));

  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void ForEach(const Stream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapperForEach<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      size, f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchKernelFix(const Stream& stream, size_t size, F f, Args&&... args) {
  KernelWrapper<<<256, 256, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}
}  // namespace rayjoin
#endif  // RAYJOIN_LAUNCHER_H
