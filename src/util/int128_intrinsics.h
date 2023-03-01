#ifndef RAYJOIN_INT128_INTRINSICS_H
#define RAYJOIN_INT128_INTRINSICS_H
#include <cuda.h>

__device__ __forceinline__ __int128 neg_int128(const __int128 x) {
  return (~x) + 1;
}

__device__ __forceinline__ __int128 add_int128(const __int128& x,
                                               const __int128& y) {
  __int128 res;
#if 0
        auto *vec_res = reinterpret_cast<longlong2 *>(&res);
        auto *vec_x = reinterpret_cast<const longlong2 *>(&x);
        auto *vec_y = reinterpret_cast<const longlong2 *>(&y);
        asm(  "add.cc.s64    %0, %2, %4;\n\t"
              "addc.s64      %1, %3, %5;\n\t"
                : "=l" (vec_res->x), "=l" (vec_res->y)
                : "l" (vec_x->x), "l" (vec_x->y),
        "l" (vec_y->x), "l" (vec_y->y));
#endif
  auto* res_vec = reinterpret_cast<int4*>(&res);
  auto* vec_x = reinterpret_cast<const int4*>(&x);
  auto* vec_y = reinterpret_cast<const int4*>(&y);

  asm("add.cc.s32      %0, %4, %8;\n\t"
      "addc.cc.s32     %1, %5, %9;\n\t"
      "addc.cc.s32     %2, %6, %10;\n\t"
      "addc.s32        %3, %7, %11;\n\t"
      : "=r"(res_vec->x), "=r"(res_vec->y), "=r"(res_vec->z), "=r"(res_vec->w)
      : "r"(vec_x->x), "r"(vec_x->y), "r"(vec_x->z), "r"(vec_x->w),
        "r"(vec_y->x), "r"(vec_y->y), "r"(vec_y->z), "r"(vec_y->w));

  return res;
}

__device__ __forceinline__ __int128 mul_int128(const __int128& x, int64_t y) {
  int64_t lo = (int64_t) x;
  int64_t hi = x >> 64;
  int64_t res_lo = lo * y;
  int64_t res_hi = __mul64hi(lo, y) + hi * y;

  return ((__int128) res_hi << 64) | res_lo;
}

#endif  // RAYJOIN_INT128_INTRINSICS_H
