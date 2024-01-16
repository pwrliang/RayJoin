

#ifndef RAYJOIN_SHARED_ARRAY_H
#define RAYJOIN_SHARED_ARRAY_H
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "util/util.h"
#include "util/stream.h"

namespace rayjoin {
template <typename T>
class SharedArray {
  static_assert(std::is_pod<T>::value, "Unsupported datatype");

 public:
  using device_t = thrust::device_vector<T>;
  using host_t = pinned_vector<T>;

  SharedArray() = default;

  explicit SharedArray(size_t size) {
    d_buffer_.resize(size);
    h_buffer_.resize(size);
  }

  void resize(size_t size) {
    d_buffer_.resize(size);
    h_buffer_.resize(size);
  }

  void set(size_t idx, const T& t) { d_buffer_[idx] = t; }

  void set(size_t idx, const T& t, const Stream& stream) {
    h_buffer_[idx] = t;
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(h_buffer_.data()),
                               sizeof(T), cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  void fill(const T& t) {
    auto size = h_buffer_.size();
    thrust::fill_n(h_buffer_.data(), size, t);
    d_buffer_ = h_buffer_;
  }

  void fill(const T& t, const Stream& stream) {
    auto size = h_buffer_.size();

    thrust::fill_n(h_buffer_.data(), size, t);
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(h_buffer_.data()),
                               sizeof(T) * size, cudaMemcpyHostToDevice,
                               stream.cuda_stream()));
  }

  typename thrust::device_vector<T>::reference get(size_t idx) {
    return d_buffer_[idx];
  }

  typename thrust::device_vector<T>::const_reference get(size_t idx) const {
    return d_buffer_[idx];
  }

  T get(size_t idx, const Stream& stream) const {
    CUDA_CHECK(cudaMemcpyAsync(
        const_cast<T*>((thrust::raw_pointer_cast(h_buffer_.data()) + idx)),
        thrust::raw_pointer_cast(d_buffer_.data()) + idx, sizeof(T),
        cudaMemcpyDeviceToHost, stream.cuda_stream()));
    stream.Sync();
    return h_buffer_[idx];
  }

  const host_t& get(const Stream& stream) const {
    CUDA_CHECK(cudaMemcpyAsync(
        const_cast<T*>(thrust::raw_pointer_cast(h_buffer_.data())),
        thrust::raw_pointer_cast(d_buffer_.data()),
        sizeof(T) * d_buffer_.size(), cudaMemcpyDeviceToHost,
        stream.cuda_stream()));
    stream.Sync();
    return h_buffer_;
  }

  host_t& get(const Stream& stream) {
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(h_buffer_.data()),
                               thrust::raw_pointer_cast(d_buffer_.data()),
                               sizeof(T) * d_buffer_.size(),
                               cudaMemcpyDeviceToHost, stream.cuda_stream()));
    stream.Sync();
    return h_buffer_;
  }

  T* data() { return thrust::raw_pointer_cast(d_buffer_.data()); }

  const T* data() const { return thrust::raw_pointer_cast(d_buffer_.data()); }

  T* data(size_t idx) {
    return thrust::raw_pointer_cast(d_buffer_.data()) + idx;
  }

  const T* data(size_t idx) const {
    return thrust::raw_pointer_cast(d_buffer_.data()) + idx;
  }

  void Assign(const SharedArray<T>& rhs) {
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_buffer_.data()),
                          thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                          sizeof(T), cudaMemcpyDefault));
  }

  void Assign(const SharedArray<T>& rhs, const Stream& stream) {
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(d_buffer_.data()),
                               thrust::raw_pointer_cast(rhs.d_buffer_.data()),
                               sizeof(T), cudaMemcpyDefault,
                               stream.cuda_stream()));
  }

  void Swap(SharedArray<T>& rhs) { d_buffer_.swap(rhs.d_buffer_); }

 private:
  device_t d_buffer_;
  host_t h_buffer_;
};
}
#endif  // RAYJOIN_SHARED_ARRAY_H
