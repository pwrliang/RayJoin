#ifndef UTIL_QUEUE_H
#define UTIL_QUEUE_H
#include <cooperative_groups.h>

#include "util/array_view.h"
#include "util/shared_value.h"
#include "util/stream.h"
namespace rayjoin {
namespace dev {
template <typename T, typename SIZE_T>
class Queue;

template <typename T>
class Queue<T, uint32_t> {
 public:
  using value_type = T;

  Queue() = default;

  DEV_HOST explicit Queue(const ArrayView<T>& data, uint32_t* last_pos)
      : data_(data), last_pos_(last_pos) {}

  DEV_INLINE void Append(const T& item) {
    auto allocation = atomicAdd(last_pos_, 1);
    assert(allocation < data_.size());
    data_[allocation] = item;
  }

  DEV_INLINE void AppendWarp(const T& item) {
    auto g = cooperative_groups::coalesced_threads();
    uint32_t warp_res;

    if (g.thread_rank() == 0) {
      warp_res = atomicAdd(last_pos_, g.size());
    }
    auto begin = g.shfl(warp_res, 0) + g.thread_rank();
    assert(begin < data_.size());
    data_[begin] = item;
  }

  DEV_INLINE void Clear() const { *last_pos_ = 0; }

  DEV_INLINE T& operator[](uint32_t i) { return data_[i]; }

  DEV_INLINE const T& operator[](uint32_t i) const { return data_[i]; }

  DEV_INLINE uint32_t size() const { return *last_pos_; }

  DEV_INLINE void Swap(Queue<T, uint32_t>& rhs) {
    data_.Swap(rhs.data_);
    thrust::swap(last_pos_, rhs.last_pos_);
  }

 private:
  ArrayView<T> data_;
  uint32_t* last_pos_{};
};

template <typename T>
class Queue<T, uint64_t> {
  static_assert(sizeof(unsigned long long int) ==  // NOLINT(runtime/int)
                sizeof(uint64_t));

 public:
  Queue() = default;

  DEV_HOST explicit Queue(const ArrayView<T>& data, uint64_t* last_pos)
      : data_(data), last_pos_(last_pos) {}

  DEV_INLINE void Append(const T& item) {
    auto allocation =
        atomicAdd((unsigned long long int*) last_pos_,  // NOLINT(runtime/int)
                  1);
    assert(allocation < data_.size());
    data_[allocation] = item;
  }

  DEV_INLINE void AppendWarp(const T& item) {
    auto g = cooperative_groups::coalesced_threads();
    uint64_t warp_res;

    if (g.thread_rank() == 0) {
      warp_res =
          atomicAdd((unsigned long long int*) last_pos_,  // NOLINT(runtime/int)
                    g.size());
    }
    auto begin = g.shfl(warp_res, 0) + g.thread_rank();
    assert(begin < data_.size());
    data_[begin] = item;
  }

  DEV_INLINE void Clear() const { *last_pos_ = 0; }

  DEV_INLINE T& operator[](uint64_t i) { return data_[i]; }

  DEV_INLINE const T& operator[](uint64_t i) const { return data_[i]; }

  DEV_INLINE uint64_t size() const { return *last_pos_; }

  DEV_INLINE void Swap(Queue<T, uint64_t>& rhs) {
    data_.Swap(rhs.data_);
    thrust::swap(last_pos_, rhs.last_pos_);
  }

 private:
  ArrayView<T> data_;
  uint64_t* last_pos_{};
};

}  // namespace dev

template <typename T, typename SIZE_T = uint32_t>
class Queue {
 public:
  using value_type = T;
  using device_t = dev::Queue<T, SIZE_T>;

  void Init(SIZE_T capacity) {
    data_.resize(capacity);
    counter_.set(0);
  }

  void Clear() { counter_.set(0); }

  void Clear(const Stream& stream) { counter_.set(0, stream); }

  size_t size() const { return counter_.get(); }

  size_t size(const Stream& stream) const { return counter_.get(stream); }

  T* data() { return thrust::raw_pointer_cast(data_.data()); }

  const T* data() const { return thrust::raw_pointer_cast(data_.data()); }

  void CopyTo(thrust::host_vector<T>& out) {
    auto s = size();
    out.resize(s);
    out.assign(data_.begin(), data_.begin() + s);
  }

  device_t DeviceObject() {
    return device_t(ArrayView<T>(data_), counter_.data());
  }

  void Swap(Queue<T, SIZE_T>& rhs) {
    data_.swap(rhs.data_);
    counter_.Swap(rhs.counter_);
  }

 private:
  thrust::device_vector<T> data_;
  SharedValue<SIZE_T> counter_{};
};
}  // namespace rayjoin
#endif  // UTIL_QUEUE_H
