#ifndef RAYJOIN_SHARED_VALUE_H
#define RAYJOIN_SHARED_VALUE_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "util/stream.h"
#include "util/util.h"

namespace rayjoin {
template <typename T>
class SharedValue {
  static_assert(std::is_pod<T>::value, "Unsupported datatype");

 public:
  SharedValue() {
    d_buffer_.resize(1);
    h_buffer_.resize(1);
  }

  void set(const T& t) { d_buffer_[0] = t; }

  void set(const T& t, const Stream& stream) {
    h_buffer_[0] = t;
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(h_buffer_.data()),
                               sizeof(T), cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  typename thrust::device_vector<T>::reference get() { return d_buffer_[0]; }

  typename thrust::device_vector<T>::const_reference get() const {
    return d_buffer_[0];
  }

  T get(const Stream& stream) const {
    CUDA_CHECK(cudaMemcpyAsync(
        const_cast<T*>(thrust::raw_pointer_cast(h_buffer_.data())),
        thrust::raw_pointer_cast(d_buffer_.data()), sizeof(T),
        cudaMemcpyDeviceToHost, stream.cuda_stream()));
    stream.Sync();
    return h_buffer_[0];
  }

  T* data() { return thrust::raw_pointer_cast(d_buffer_.data()); }

  const T* data() const { return thrust::raw_pointer_cast(d_buffer_.data()); }

  void Assign(const SharedValue<T>& rhs) {
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_buffer_.data()),
                          thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                          sizeof(T), cudaMemcpyDefault));
  }

  void Assign(const SharedValue<T>& rhs, const Stream& stream) {
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                               sizeof(T), cudaMemcpyDefault,
                               stream.cuda_stream()));
  }

  void Swap(SharedValue<T>& rhs) { d_buffer_.swap(rhs.d_buffer_); }

 private:
  thrust::device_vector<T> d_buffer_;
  thrust::host_vector<T, thrust::cuda::experimental::pinned_allocator<T>>
      h_buffer_;
};
}  // namespace rayjoin

#endif  // RAYJOIN_SHARED_VALUE_H
