#ifndef RAYJOIN_UTIL_XMP_H
#define RAYJOIN_UTIL_XMP_H
#include "util/u64.h"
#ifdef __CUDACC__
#include "cuda.h"
#define HD __host__ __device__
#define DV __device__
#else
#include <assert.h>
#define HD
#define DV
#endif

/**
   XMP: Extended Precision Math Primitives
   This is a collection of functions that implement math primitives
   like add-with-carry-in and multiply-add.  They are implement both
   in regular C code and PTX assembly for use in CUDA programs.
   Correction: the PTX routines dealing with the carry bit cannot be relied
   on, because the carry bit is not considered when dead code is
   eliminated.  This issue came up with the multiply-hi operation.
   The lower word multiplication are needed to compute carries into the
   upper words, but they get eliminated because their results (other than
   the carry bit) are ignored.
   Implemented as inline class methods, these will be compiled as
   inline PTX code by the CUDA complier.
   Routines with PTX or intrinsic versions:
     join64
     split64
     addCarryOut
     addCarryInOut
     addCarryIn
     addCarryOut64
     addCarryInOut64
     addCarryIn64
     subBorrowOut
     subBorrowInOut
     subBorrowIn
     multHi
     multHi64
     multAdd
     multAddCarryOut
     multAddCarryInOut
     multAddCarryIn
     multHiAdd
     multHiAddCarryOut
     multHiAddCarryInOut
     multHiAddCarryIn
     multFull
     multFull64
     multFullAdd64
     multFullAddCarryOut64
     multFullAddCarryInOut64
     countLeadingZeros
     countLeadingZeros64
 */

class XMP {
 public:
#ifdef __CUDACC__

  __device__ static u64 join64_ptx(unsigned hi, unsigned lo) {
    u64 x;
    asm("mov.b64 %0,{%1,%2};\n" : "=l"(x) : "r"(lo), "r"(hi));
    return x;
  }

  __device__ static void split64_ptx(u64 x, unsigned &hi, unsigned &lo) {
    unsigned lo_, hi_;
    asm("mov.b64 {%0,%1},%2;\n" : "=r"(lo_), "=r"(hi_) : "l"(x));
    lo = lo_;
    hi = hi_;
  }

  __device__ static u64 multFull_ptx(unsigned f1, unsigned f2) {
    u64 result;
    asm("mul.wide.u32 %0, %1, %2;\n"
        : "=l"(result)
        : "r"(f1), "r"(f2));
    return result;
  }

  __device__ static void multFull64_intrinsic(u64 f1, u64 f2,
                                              u64 &resultHi, u64 &resultLo) {
    resultLo = f1 * f2;
    resultHi = __umul64hi(f1, f2);
  }

  // return (a*b) / 2^64
  __device__ static u64 multHi64_intrinsic(u64 a, u64 b) {
    return __umul64hi(a, b);
  }

  // return (a*b) / 2^32
  __device__ static unsigned multHi_intrinsic(unsigned a, unsigned b) {
    return __umulhi(a, b);
  }

  __device__ static unsigned multAdd_ptx(unsigned f1, unsigned f2, unsigned a) {
    unsigned result;
    asm("mad.lo.u32 %0, %1, %2, %3;\n"
        : "=r"(result)
        : "r"(f1), "r"(f2), "r"(a));
    return result;
  }

  __device__ static unsigned multHiAdd_ptx
      (unsigned f1, unsigned f2, unsigned a) {
    unsigned result;
    asm("mad.hi.u32 %0, %1, %2, %3;\n"
        : "=r"(result)
        : "r"(f1), "r"(f2), "r"(a));
    return result;
  }

  __device__ static int countLeadingZeros_intrinsic(unsigned x) {
    return __clz(x);
  }

  __device__ static int countLeadingZeros64_intrinsic(u64 x) {
    return __clzll(x);
  }

  // {hi,lo} += a*b
  __device__ static void multFullAdd64_ptx(u64 a, u64 b,
                                           u64 &hi, u64 &lo) {
    asm("{"
        ".reg .u32 d<4>;\n\t"
        ".reg .u64 m<2>;\n\t"
        ".reg .u32 x<4>;\n\t"
        "mov.b64 {d0,d1},%0;\n\t"
        "mov.b64 {d2,d3},%1;\n\t"

        "mul.lo.u64 m0, %2, %3;\n\t"
        "mul.hi.u64 m1, %2, %3;\n\t"
        "mov.b64 {x0,x1},m0;\n\t"
        "mov.b64 {x2,x3},m1;\n\t"

        "add.cc.u32  d0, d0, x0;\n\t"
        "addc.cc.u32 d1, d1, x1;\n\t"
        "addc.cc.u32 d2, d2, x2;\n\t"
        "addc.u32    d3, d3, x3;\n\t"

        "mov.b64     %0,{d0,d1};\n\t"
        "mov.b64     %1,{d2,d3};\n\t"
        "}\n\t"
        : "+l"(lo), "+l"(hi)
        : "l"(a), "l"(b));
  }

#endif // __CUDACC__

  // combine u[0] and u[1] into a 64-bit value, where u[0] is the less
  // significant one
  HD static u64 join64(const unsigned *u) {
    // return ((u64)u[1] << 32) | u[0];
    return join64(u[1], u[0]);
  }

  // all these silly #ifdef's are necessary because I'd like to be able
  // to compile the code without CUDA support, and intrinsic functions cannot
  // be called host functions.

   HD static u64 join64(unsigned hi, unsigned lo) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    return ((u64)hi << 32) | lo;
#else
    return join64_ptx(hi, lo);
#endif
  }

  HD static void split64(u64 x, unsigned &hi, unsigned &lo) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    lo = (unsigned)x;
    hi = (unsigned)(x>>32);
#else
    split64_ptx(x, hi, lo);
#endif
  }

  HD static unsigned addCarryOut(unsigned a, unsigned b, int &carry) {
    unsigned result = a + b;
    carry = (result < a);
    return result;
  }


  HD static unsigned addCarryInOut(unsigned a, unsigned b, int &carry) {
    unsigned result = a + b + carry;
    if (carry)
      carry = (result <= a);
    else
      carry = (result < a);
    return result;
  }

  HD static unsigned addCarryIn(unsigned a, unsigned b, int &carry) {
    return a + b + carry;
  }

  HD static u64 addCarryOut64(u64 a, u64 b, int &carry) {
    u64 result = a + b;
    carry = (result < a);
    return result;
  }

  HD static u64 addCarryInOut64(u64 a, u64 b, int &carry) {
    u64 result = a + b + carry;
    if (carry)
      carry = (result <= a);
    else
      carry = (result < a);
    return result;
  }

  HD static u64 addCarryIn64(u64 a, u64 b, int &carry) {
    return a + b + carry;
  }

  HD static unsigned subBorrowOut(unsigned a, unsigned b, int &carry) {
    unsigned result = a - b;
    carry = (result > a);
    return result;
  }

  HD static unsigned subBorrowInOut(unsigned a, unsigned b, int &carry) {
    unsigned result = a - b - carry;
    if (carry)
      carry = (result >= a);
    else
      carry = (result > a);
    return result;
  }

  HD static unsigned subBorrowIn(unsigned a, unsigned b, int &carry) {
    return a - b - carry;
  }

  HD static unsigned multHi(unsigned f1, unsigned f2) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    return (unsigned) (((u64)f1 * f2) >> 32);
#else
    return multHi_intrinsic(f1, f2);
#endif
  }

  HD static u64 multHi64(u64 f1, u64 f2) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    /*
    u64 lo = (a & LO_MASK) * (unsigned)b;
    u64 mid = lo >> 32;
    int carry = addWithCarry64(mid, (a & LO_MASK) * (b >> 32));
    carry += addWithCarry64(mid, (b & LO_MASK) * (a >> 32));
    // result_lo = (lo & LO_MASK) | (mid << 32);
    return (mid >> 32) + (a >> 32) * (b >> 32) + ((u64)carry << 32);
    */
    unsigned a, b, c, d, lo, hi, unused;
    int carry;

    split64(f1, a, b);
    split64(f2, c, d);

    unused = multHi(b, d);
    unused = multAddCarryOut(a, d, unused, carry);
    lo = multHiAddCarryIn(a, d, 0, carry);
    /* unused = */ multAddCarryOut(b, c, unused, carry);
    lo = multHiAddCarryInOut(b, c, lo, carry);
    hi = addCarryIn(0, 0, carry);
    lo = multAddCarryOut(a, c, lo, carry);
    hi = multHiAddCarryIn(a, c, hi, carry);

    // or use multFull64

    return join64(hi, lo);
#else
    return multHi64_intrinsic(f1, f2);
#endif
  }

  HD static unsigned multAdd(unsigned f1, unsigned f2, unsigned a) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    return f1*f2 + a;
#else
    return multAdd_ptx(f1, f2, a);
#endif
  }

  HD static unsigned multAddCarryOut(unsigned f1, unsigned f2, unsigned a,
                                     int &carry) {
    return addCarryOut(f1*f2, a, carry);
  }

  HD static unsigned multAddCarryInOut(unsigned f1, unsigned f2, unsigned a,
                                       int &carry) {
    return addCarryInOut(f1*f2, a, carry);
  }

  HD static unsigned multAddCarryIn(unsigned f1, unsigned f2, unsigned a,
                                    int &carry) {
    return addCarryIn(f1*f2, a, carry);
  }

  HD static unsigned multHiAdd(unsigned f1, unsigned f2, unsigned a) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    return multHi(f1, f2) + a;
#else
    return multHiAdd_ptx(f1, f2, a);
#endif
  }

  HD static unsigned multHiAddCarryOut(unsigned f1, unsigned f2, unsigned a,
                                       int &carry) {
    return addCarryOut(multHi(f1, f2), a, carry);
  }

  HD static unsigned multHiAddCarryInOut(unsigned f1, unsigned f2, unsigned a,
                                         int &carry) {
    return addCarryInOut(multHi(f1, f2), a, carry);
  }

  HD static unsigned multHiAddCarryIn(unsigned f1, unsigned f2, unsigned a,
                                      int &carry) {
    return addCarryIn(multHi(f1, f2), a, carry);
  }


  HD static u64 multFull(unsigned f1, unsigned f2) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    return (u64)f1 * f2;
#else
    return multFull_ptx(f1, f2);
#endif
  }


  HD static void multFull(unsigned f1, unsigned f2,
                          unsigned &resultHi, unsigned &resultLo) {
    u64 result = multFull(f1, f2);
    split64(result, resultHi, resultLo);
  }


  HD static void multFull64(u64 f1, u64 f2, u64 &resultHi, u64 &resultLo) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    unsigned a, b, c, d, result0, result1, result2, result3;
    int carry;

    split64(f1, a, b);
    split64(f2, c, d);

    result0 = b * d;
    result1 = multHi(b, d);
    result1 = multAddCarryOut(a, d, result1, carry);
    result2 = multHiAddCarryIn(a, d, 0, carry);
    result1 = multAddCarryOut(b, c, result1, carry);
    resultLo = join64(result1, result0);

    result2 = multHiAddCarryInOut(b, c, result2, carry);
    result3 = addCarryIn(0, 0, carry);
    result2 = multAddCarryOut(a, c, result2, carry);
    result3 = multHiAddCarryIn(a, c, result3, carry);

    resultHi = join64(result3, result2);
#else
    multFull64_intrinsic(f1, f2, resultHi, resultLo);
#endif
  }

  // add in place
  HD static void multFullAdd64(u64 f1, u64 f2,
                               u64 &resultHi, u64 &resultLo) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    u64 tmpLo, tmpHi;
    multFull64(f1, f2, tmpHi, tmpLo);
    int carry;
    resultLo = addCarryOut64(tmpLo, resultLo, carry);
    resultHi = addCarryIn64(tmpHi, resultHi, carry);
#else
    multFullAdd64_ptx(f1, f2, resultHi, resultLo);
#endif
  }

  HD static void multFullAddCarryOut64(u64 f1, u64 f2,
                                       u64 &resultHi, u64 &resultLo,
                                       int &carry) {
    u64 tmpLo, tmpHi;
    multFull64(f1, f2, tmpHi, tmpLo);
    resultLo = addCarryOut64(tmpLo, resultLo, carry);
    resultHi = addCarryInOut64(tmpHi, resultHi, carry);
  }

  HD static void multFullAddCarryInOut64(u64 f1, u64 f2,
                                         u64 &resultHi, u64 &resultLo,
                                         int &carry) {
    u64 tmpLo, tmpHi;
    // multFull64_intrinsic must not change carry flag
    multFull64(f1, f2, tmpHi, tmpLo);
    resultLo = addCarryInOut64(tmpLo, resultLo, carry);
    resultHi = addCarryInOut64(tmpHi, resultHi, carry);
  }


  HD static int countLeadingZeros(unsigned x) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    unsigned y, n;
    n = 32;
    y = x >> 16; if (y != 0) {n-=16; x=y;}
    y = x >>  8; if (y != 0) {n-= 8; x=y;}
    y = x >>  4; if (y != 0) {n-= 4; x=y;}
    y = x >>  2; if (y != 0) {n-= 2; x=y;}
    y = x >>  1; if (y != 0) return n - 2;
    return n - x;
#else
    return countLeadingZeros_intrinsic(x);
#endif
  }

  HD static int countLeadingZeros64(u64 x) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    unsigned y, n;
    n = 64;
    y = x >> 32; if (y != 0) {n-=32; x=y;}
    y = x >> 16; if (y != 0) {n-=16; x=y;}
    y = x >>  8; if (y != 0) {n-= 8; x=y;}
    y = x >>  4; if (y != 0) {n-= 4; x=y;}
    y = x >>  2; if (y != 0) {n-= 2; x=y;}
    y = x >>  1; if (y != 0) return n - 2;
    return n - x;
#else
    return countLeadingZeros64_intrinsic(x);
#endif
  }

  HD static unsigned highestSetBit(unsigned x) {
    return 1u << (31 - countLeadingZeros(x));
  }

  HD static u64 highestSetBit64(u64 x) {
    return 1ull << (63 - countLeadingZeros64(x));
  }

  // format an unsigned to 8 hex chars
  static HD void toHex(char buf[8], unsigned x) {
    for (int i=7; i >= 0; i--) {
      buf[i] = "0123456789abcdef"[x & 15];
      x >>= 4;
    }
  }
};
#endif  // RAYJOIN_UTIL_XMP_H
