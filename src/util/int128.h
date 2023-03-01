#ifndef RAYJOIN_UTIL_INT128
#define RAYJOIN_UTIL_INT128

/*
  128-bit extra long integer class
  Using four 32-bit integers internally.

  Ed Karrels, May 2012
*/

// http://tauday.com/
// http://stackoverflow.com/questions/6162140/128-bit-integer-on-cuda
// http://stackoverflow.com/questions/6659414/efficient-128-bit-addition-using-carry-flag?rq=1
// http://software.intel.com/en-us/forums/topic/278561

#include <cassert>
#include <cctype>
#include <climits>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "util/xmp.h"

#if UINT_MAX != 4294967295
#error Assume 32 bit unsigned integer
#endif

#define LO_MASK 0x00000000FFFFFFFFLL
#define HI_MASK 0xFFFFFFFF00000000LL

class Int128;  // forward declaration

namespace std {
template <>
struct is_integral<Int128> : public true_type {};

std::ostream& operator<<(std::ostream& o, Int128 x);
}  // namespace std

class Int128 {
 public:
#if NVCC_VERSION == 50
  // for some reason, this slows down a bit with the 5.0 compiler
  // if I remove the union
  union {
    struct {
      uint64_t unused0, unused1;
    };
    struct {
      unsigned dataword0, dataword1, dataword2, dataword3;
    };
  };
#else
  unsigned dataword0, dataword1, dataword2, dataword3;
#endif

  HD Int128() {}

  HD Int128(int32_t init) {
    setPart0(init);
    setPart1(0);
  }

  HD Int128(int64_t init) {
    setPart0(init);
    setPart1(0);
  }

  HD Int128(uint32_t init) {
    setPart0(init);
    setPart1(0);
  }

  HD Int128(uint64_t init) {
    setPart0(init);
    setPart1(0);
  }

  HD Int128(uint64_t hi, uint64_t lo) {
    setPart0(lo);
    setPart1(hi);
  }

  HD Int128(const __int128& i) {
    setPart0((uint64_t) i);
    setPart1((uint64_t) (i >> 64));
  }

  HD Int128(unsigned w3, unsigned w2, unsigned w1, unsigned w0)
      : dataword0(w0), dataword1(w1), dataword2(w2), dataword3(w3) {}

  HD Int128(const Int128& other)
      : dataword0(other.dataword0),
        dataword1(other.dataword1),
        dataword2(other.dataword2),
        dataword3(other.dataword3) {}

  HD uint64_t getPart1() const { return XMP::join64(dataword3, dataword2); }
  HD uint64_t getPart0() const { return XMP::join64(dataword1, dataword0); }

  HD void setPart1(uint64_t x) {
    unsigned hi, lo;
    XMP::split64(x, hi, lo);
    setWord2(lo);
    setWord3(hi);
  }

  HD void setPart0(uint64_t x) {
    unsigned hi, lo;
    XMP::split64(x, hi, lo);
    setWord0(lo);
    setWord1(hi);
  }

  HD uint64_t to_uint64_t() const { return getPart0(); }

  HD unsigned& word0() { return dataword0; }
  HD unsigned getWord0() const { return dataword0; }
  HD void setWord0(unsigned x) { dataword0 = x; }

  HD unsigned& word1() { return dataword1; }
  HD unsigned getWord1() const { return dataword1; }
  HD void setWord1(unsigned x) { dataword1 = x; }

  HD unsigned& word2() { return dataword2; }
  HD unsigned getWord2() const { return dataword2; }
  HD void setWord2(unsigned x) { dataword2 = x; }

  HD unsigned& word3() { return dataword3; }
  HD unsigned getWord3() const { return dataword3; }
  HD void setWord3(unsigned x) { dataword3 = x; }

  bool from_hex(const char* str) {
    // skip leading whitespace
    while (isspace(*str))
      str++;

    // skip optional "0x" prefix
    if (str[0] == '0' && tolower(str[1]) == 'x')
      str += 2;

    if (!isxdigit(*str))
      return false;

    Int128 result((uint64_t) 0);

    for (; isxdigit(*str); str++) {
      int32_t digit;
      if (isdigit(*str)) {
        digit = *str - '0';
      } else {
        digit = tolower(*str) - 'a' + 10;
      }
      result = result * 16 + digit;
    }

    *this = result;

    return true;
  }

  HD const char* toString(char buf[33]) const {
    XMP::toHex(buf, getWord3());
    XMP::toHex(buf + 8, getWord2());
    XMP::toHex(buf + 16, getWord1());
    XMP::toHex(buf + 24, getWord0());
    buf[32] = '\0';
    return buf;
  }

  // overload equality
  HD bool operator==(Int128 x) const {
    return getWord0() == x.getWord0() && getWord1() == x.word1() &&
           getWord2() == x.word2() && getWord3() == x.word3();
  }

  HD bool operator==(uint64_t x) const {
    return getPart0() == x && getWord2() == 0 && getWord3() == 0;
  }

  HD bool operator!=(Int128 x) const {
    return getWord0() != x.getWord0() || getWord1() != x.getWord1() ||
           getWord2() != x.getWord2() || getWord3() != x.getWord3();
  }

  HD bool operator!=(uint64_t x) const {
    return getPart0() != x || getWord2() != 0 || getWord3() != 0;
  }

  HD bool operator<(Int128 x) const {
    if (getWord3() == x.getWord3()) {
      if (getWord2() == x.getWord2()) {
        if (getWord1() == x.getWord1()) {
          return getWord0() < x.getWord0();
        } else {
          return getWord1() < x.getWord1();
        }
      } else {
        return getWord2() < x.getWord2();
      }
    } else {
      return getWord3() < x.getWord3();
    }
  }

  HD bool operator>(Int128 that) const { return that < *this; }

  HD bool operator<=(Int128 that) const { return !(that < *this); }

  HD bool operator>=(Int128 that) const { return !(*this < that); }

  // cast to double - more trouble than it's worth
  HD operator double() const {
    double result = (double) getPart0();
    if (getPart1()) {
      // 1<<64
      // result += 18446744073709551616.0 * getPart1();
      result += (double) ULLONG_MAX * getPart1();
    }
    return result;
  }

  HD double toFraction() const {
    return ((double) *this) / 3.4028236692093846e+38;
  }

  // overload shift operators

  HD Int128 operator<<(unsigned s) const {
    Int128 result;
    s &= 127;  // modulo 128
    if (s == 0)
      return *this;
    if (s >= 64) {
      result.setPart0(0);
      result.setPart1(getPart0() << (s - 64));
    } else {
      result.setPart0(getPart0() << s);
      result.setPart1((getPart1() << s) | (getPart0() >> (64 - s)));
    }
    return result;
  }

  HD Int128 operator>>(unsigned s) const {
    Int128 result;
    s &= 127;  // modulo 128
    if (s == 0)
      return *this;
    if (s >= 64) {
      result.setPart1(0);
      result.setPart0(getPart1() >> (s - 64));
    } else {
      result.setPart1(getPart1() >> s);
      result.setPart0((getPart0() >> s) | (getPart1() << (64 - s)));
    }
    return result;
  }

  HD Int128 operator>>=(unsigned s) { return *this = *this >> s; }

  HD Int128 operator<<=(unsigned s) { return *this = *this << s; }

  // overload bitwise operators

  HD Int128 operator|(Int128 x) const {
    return Int128(getWord3() | x.getWord3(), getWord2() | x.getWord2(),
                  getWord1() | x.getWord1(), getWord0() | x.getWord0());
  }

  HD Int128 operator&(Int128 x) const {
    return Int128(getWord3() & x.getWord3(), getWord2() & x.getWord2(),
                  getWord1() & x.getWord1(), getWord0() & x.getWord0());
  }

  HD Int128 operator&=(Int128 x) { return *this = *this & x; }

  HD Int128 operator~() const {
    return Int128(~getWord3(), ~getWord2(), ~getWord1(), ~getWord0());
  }

  HD Int128 operator-() const { return ~(*this) + 1; }

  HD int countLeadingZeros() const {
    if (getWord3()) {
      return XMP::countLeadingZeros(getWord3());
    } else if (getWord2()) {
      return 32 + XMP::countLeadingZeros(getWord2());
    } else if (getWord1()) {
      return 64 + XMP::countLeadingZeros(getWord1());
    } else {
      return 96 + XMP::countLeadingZeros(getWord0());
    }
  }

  HD float toFloat() const {
    return getWord3() * 79228162514264337593543950336.0f +  // 2^96
           getWord2() * 18446744073709551616.0f +           // 2^64
           getWord1() * 4294967296.0f +                     // 2^32
           getWord0();
  }

#ifdef __CUDACC__
  __device__ static Int128 add_128_128_asm(Int128 a, Int128 b) {
    Int128 result;
    asm("{\n\t"
        "add.cc.u32    %0, %4, %8;\n\t"
        "addc.cc.u32   %1, %5, %9;\n\t"
        "addc.cc.u32   %2, %6, %10;\n\t"
        "addc.u32      %3, %7, %11;\n\t"
        "}\n"
        : "=r"(result.word0()), "=r"(result.word1()), "=r"(result.word2()),
          "=r"(result.word3())
        : "r"(a.word0()), "r"(a.word1()), "r"(a.word2()), "r"(a.word3()),
          "r"(b.word0()), "r"(b.word1()), "r"(b.word2()), "r"(b.word3()));
    return result;
  }

  __device__ static Int128 add_128_64_asm(Int128 a, uint64_t b) {
    Int128 result;
    asm("{\n\t"
        ".reg .u32 sb<2>;\n\t"
        "mov.b64   {sb0,sb1}, %8;\n\t"
        "add.cc.u32    %0, %4, sb0;\n\t"
        "addc.cc.u32   %1, %5, sb1;\n\t"
        "addc.cc.u32   %2, %6, 0;\n\t"
        "addc.u32      %3, %7, 0;\n\t"
        "}\n"
        : "=r"(result.word0()), "=r"(result.word1()), "=r"(result.word2()),
          "=r"(result.word3())
        : "r"(a.word0()), "r"(a.word1()), "r"(a.word2()), "r"(a.word3()),
          "l"(b));
    return result;
  }

  __device__ static Int128 sub_128_128_asm(Int128 a, Int128 b) {
    Int128 result;
    asm("{\n\t"
        "sub.cc.u32    %0, %4, %8;\n\t"
        "subc.cc.u32   %1, %5, %9;\n\t"
        "subc.cc.u32   %2, %6, %10;\n\t"
        "subc.u32      %3, %7, %11;\n\t"
        "}\n"
        : "=r"(result.word0()), "=r"(result.word1()), "=r"(result.word2()),
          "=r"(result.word3())
        : "r"(a.word0()), "r"(a.word1()), "r"(a.word2()), "r"(a.word3()),
          "r"(b.word0()), "r"(b.word1()), "r"(b.word2()), "r"(b.word3()));
    return result;
  }

  __device__ static Int128 sub_128_64_asm(Int128 a, uint64_t b) {
    Int128 result;
    asm("{\n\t"
        ".reg .u32 sb<2>;\n\t"
        "mov.b64   {sb0,sb1}, %8;\n\t"
        "sub.cc.u32    %0, %4, sb0;\n\t"
        "subc.cc.u32   %1, %5, sb1;\n\t"
        "subc.cc.u32   %2, %6, 0;\n\t"
        "subc.u32      %3, %7, 0;\n\t"
        "}\n"
        : "=r"(result.word0()), "=r"(result.word1()), "=r"(result.word2()),
          "=r"(result.word3())
        : "r"(a.word0()), "r"(a.word1()), "r"(a.word2()), "r"(a.word3()),
          "l"(b));
    return result;
  }

  __device__ static void mult128_asm(Int128 a, Int128 b, Int128& result_hi,
                                     Int128& result_lo) {
    // write_mult_asm.py 2 4 full 'a.word%d()' 'b.word%d()'
    // 'result_lo.word%d(),result_hi.word%d()'
    asm("{\n\t"
        ".reg .u32 tmp;\n\t"
        "mul.lo.u32       %0, %8, %12;\n\t"
        "mul.lo.u32       %1, %8, %13;\n\t"
        "mul.lo.u32       %2, %8, %14;\n\t"
        "mul.lo.u32       %3, %8, %15;\n\t"

        "mul.hi.u32       tmp, %8, %12;\n\t"
        "add.cc.u32       %1, %1, tmp;\n\t"
        "mul.hi.u32       tmp, %8, %13;\n\t"
        "addc.cc.u32      %2, %2, tmp;\n\t"
        "mul.hi.u32       tmp, %8, %14;\n\t"
        "addc.cc.u32      %3, %3, tmp;\n\t"
        "mul.hi.u32       tmp, %8, %15;\n\t"
        "addc.u32         %4, 0, tmp;\n\t"

        "mul.lo.u32       tmp, %9, %12;\n\t"
        "add.cc.u32       %1, %1, tmp;\n\t"
        "mul.lo.u32       tmp, %9, %13;\n\t"
        "addc.cc.u32      %2, %2, tmp;\n\t"
        "mul.lo.u32       tmp, %9, %14;\n\t"
        "addc.cc.u32      %3, %3, tmp;\n\t"
        "mul.lo.u32       tmp, %9, %15;\n\t"
        "addc.u32         %4, %4, tmp;\n\t"

        "mul.hi.u32       tmp, %9, %12;\n\t"
        "add.cc.u32       %2, %2, tmp;\n\t"
        "mul.hi.u32       tmp, %9, %13;\n\t"
        "addc.cc.u32      %3, %3, tmp;\n\t"
        "mul.hi.u32       tmp, %9, %14;\n\t"
        "addc.cc.u32      %4, %4, tmp;\n\t"
        "mul.hi.u32       tmp, %9, %15;\n\t"
        "addc.u32         %5, 0, tmp;\n\t"

        "mul.lo.u32       tmp, %10, %12;\n\t"
        "add.cc.u32       %2, %2, tmp;\n\t"
        "mul.lo.u32       tmp, %10, %13;\n\t"
        "addc.cc.u32      %3, %3, tmp;\n\t"
        "mul.lo.u32       tmp, %10, %14;\n\t"
        "addc.cc.u32      %4, %4, tmp;\n\t"
        "mul.lo.u32       tmp, %10, %15;\n\t"
        "addc.u32         %5, %5, tmp;\n\t"

        "mul.hi.u32       tmp, %10, %12;\n\t"
        "add.cc.u32       %3, %3, tmp;\n\t"
        "mul.hi.u32       tmp, %10, %13;\n\t"
        "addc.cc.u32      %4, %4, tmp;\n\t"
        "mul.hi.u32       tmp, %10, %14;\n\t"
        "addc.cc.u32      %5, %5, tmp;\n\t"
        "mul.hi.u32       tmp, %10, %15;\n\t"
        "addc.u32         %6, 0, tmp;\n\t"

        "mul.lo.u32       tmp, %11, %12;\n\t"
        "add.cc.u32       %3, %3, tmp;\n\t"
        "mul.lo.u32       tmp, %11, %13;\n\t"
        "addc.cc.u32      %4, %4, tmp;\n\t"
        "mul.lo.u32       tmp, %11, %14;\n\t"
        "addc.cc.u32      %5, %5, tmp;\n\t"
        "mul.lo.u32       tmp, %11, %15;\n\t"
        "addc.u32         %6, %6, tmp;\n\t"

        "mul.hi.u32       tmp, %11, %12;\n\t"
        "add.cc.u32       %4, %4, tmp;\n\t"
        "mul.hi.u32       tmp, %11, %13;\n\t"
        "addc.cc.u32      %5, %5, tmp;\n\t"
        "mul.hi.u32       tmp, %11, %14;\n\t"
        "addc.cc.u32      %6, %6, tmp;\n\t"
        "mul.hi.u32       tmp, %11, %15;\n\t"
        "addc.u32         %7, 0, tmp;\n\t"

        "}\n\t"
        : "=r"(result_lo.word0()), "=r"(result_lo.word1()),
          "=r"(result_lo.word2()), "=r"(result_lo.word3()),
          "=r"(result_hi.word0()), "=r"(result_hi.word1()),
          "=r"(result_hi.word2()), "=r"(result_hi.word3())
        : "r"(a.word0()), "r"(a.word1()), "r"(a.word2()), "r"(a.word3()),
          "r"(b.word0()), "r"(b.word1()), "r"(b.word2()), "r"(b.word3()));
  }

  __device__ static int atomicAddWithCarry(unsigned* result, unsigned value,
                                           int carryIn) {
    unsigned old = ::atomicAdd(result, value + carryIn);
    unsigned sum = old + value + carryIn;
    if (carryIn)
      return sum <= value;
    else
      return sum < value;
  }

  __device__ void atomicAdd(Int128 value) {
    int carry;
    carry = atomicAddWithCarry(&dataword0, value.word0(), 0);
    carry = atomicAddWithCarry(&dataword1, value.word1(), carry);
    carry = atomicAddWithCarry(&dataword2, value.word2(), carry);
    atomicAddWithCarry(&dataword3, value.word3(), carry);
  }

#endif

  HD Int128 operator+(int32_t x) const { return *this + Int128(x); }

  HD Int128 operator+(int64_t x) const { return *this + Int128(x); }

  HD Int128 operator+(uint32_t x) const { return *this + Int128(x); }

  HD Int128 operator+(uint64_t x) const { return *this + Int128(x); }

  friend HD Int128 operator+(int32_t l, const Int128& r) { return r + l; }

  friend HD Int128 operator+(int64_t l, const Int128& r) { return r + l; }

  friend HD Int128 operator+(uint32_t l, const Int128& r) { return r + l; }

  friend HD Int128 operator+(uint64_t l, const Int128& r) { return r + l; }

  HD Int128 operator+(Int128 x) const {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    Int128 result;
    int carry;
    result.setWord0(XMP::addCarryOut(getWord0(), x.word0(), carry));
    result.setWord1(XMP::addCarryInOut(getWord1(), x.word1(), carry));
    result.setWord2(XMP::addCarryInOut(getWord2(), x.word2(), carry));
    result.setWord3(XMP::addCarryIn(getWord3(), x.word3(), carry));
    return result;
#else
    return add_128_128_asm(*this, x);
#endif
  }

  HD Int128 operator+=(int32_t x) { return *this = *this + x; }

  HD Int128 operator+=(int64_t x) { return *this = *this + x; }

  HD Int128 operator+=(uint32_t x) { return *this = *this + x; }

  HD Int128 operator+=(uint64_t x) { return *this = *this + x; }

  HD Int128 operator+=(const Int128 x) { return *this = *this + x; }

  HD Int128 operator-(int32_t x) const { return *this - Int128(x); }

  HD Int128 operator-(int64_t x) const { return *this - Int128(x); }

  HD Int128 operator-(uint32_t x) const { return *this - Int128(x); }

  HD Int128 operator-(uint64_t x) const { return *this - Int128(x); }

  friend HD Int128 operator-(int32_t l, const Int128& r) {
    return Int128(l) - r;
  }

  friend HD Int128 operator-(int64_t l, const Int128& r) {
    return Int128(l) - r;
  }

  friend HD Int128 operator-(uint32_t l, const Int128& r) {
    return Int128(l) - r;
  }

  friend HD Int128 operator-(uint64_t l, const Int128& r) {
    return Int128(l) - r;
  }

  // overload subtraction operator
  HD Int128 operator-(Int128 x) const {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    Int128 result;
    int borrow;
    result.setWord0(XMP::subBorrowOut(getWord0(), x.word0(), borrow));
    result.setWord1(XMP::subBorrowInOut(getWord1(), x.word1(), borrow));
    result.setWord2(XMP::subBorrowInOut(getWord2(), x.word2(), borrow));
    result.setWord3(XMP::subBorrowIn(getWord3(), x.word3(), borrow));
    return result;
#else
    return sub_128_128_asm(*this, x);
#endif
  }

  HD Int128 operator-=(const Int128 x) { return *this = *this - x; }

  // multiplication
  // variations included:
  // 64*64 -> 128, hi(64*64)  (lo not necessary; built-in)
  // 128*64 -> 128
  // 128*128 -> 256, hi(128*128), lo(128*128)

  HD static Int128 mult64(uint64_t a, uint64_t b) {
    uint64_t tmpHi, tmpLo;
    XMP::multFull64(a, b, tmpHi, tmpLo);
    return Int128(tmpHi, tmpLo);
  }

  /*
    Full 128-bit multiply -> 256 bit result
           A B
           C D
        ------
          --BD
        --AD
        --BC
      --AC
  */

  HD static void mult128(Int128 a, Int128 b, Int128& result_hi,
                         Int128& result_lo) {
#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    int carry;
    // --BD
    result_lo = mult64(a.getPart0(), b.getPart0());

    // --AC
    result_hi = mult64(a.getPart1(), b.getPart1());

    // --AD
    Int128 tmp = mult64(a.getPart1(), b.getPart0());
    result_lo.setPart1(
        XMP::addCarryOut64(result_lo.getPart1(), tmp.getPart0(), carry));
    result_hi.setPart0(
        XMP::addCarryInOut64(result_hi.getPart0(), tmp.getPart1(), carry));
    result_hi.setPart1(XMP::addCarryIn64(result_hi.getPart1(), 0, carry));

    // --BC
    tmp = mult64(a.getPart0(), b.getPart1());
    result_lo.setPart1(
        XMP::addCarryOut64(result_lo.getPart1(), tmp.getPart0(), carry));
    result_hi.setPart0(
        XMP::addCarryInOut64(result_hi.getPart0(), tmp.getPart1(), carry));
    result_hi.setPart1(XMP::addCarryIn64(result_hi.getPart1(), 0, carry));

#else

    mult128_asm(a, b, result_hi, result_lo);

#endif
  }

  HD static Int128 mult128hi(Int128 a, Int128 b) {
    Int128 result_lo, result_hi;
    mult128(a, b, result_hi, result_lo);
    return result_hi;
  }

  HD static Int128 mult128lo(Int128 a, Int128 b) {
    Int128 result;

#if !defined(__CUDACC__) || !defined(__CUDA_ARCH__)
    result.setPart0(a.getPart0() * b.getPart0());
    result.setPart1(XMP::multHi64(a.getPart0(), b.getPart0()) +
                    a.getPart0() * b.getPart1() + a.getPart1() * b.getPart0());
#else
    result = mult128lo_asm(a, b);
#endif

    return result;
  }

#ifdef __CUDACC__
  __device__ static Int128 mult128lo_asm(Int128 a, Int128 b) {
    /*
   // write_mult_asm.py 2 4 lo 'a.word%d()' 'b.word%d()' 'result.word%d()'
   Int128 result;
   asm( "{\n\t"
    ".reg .u32 tmp;\n\t"
    "mul.lo.u32       %0, %4, %8;\n\t"
    "mul.lo.u32       %1, %4, %9;\n\t"
    "mul.lo.u32       %2, %4, %10;\n\t"
    "mul.lo.u32       %3, %4, %11;\n\t"

       "mul.hi.u32       tmp, %4, %8;\n\t"
       "add.cc.u32       %1, %1, tmp;\n\t"
       "mul.hi.u32       tmp, %4, %9;\n\t"
       "addc.cc.u32      %2, %2, tmp;\n\t"
       "mul.hi.u32       tmp, %4, %10;\n\t"
       "addc.u32         %3, %3, tmp;\n\t"

       "mul.lo.u32       tmp, %5, %8;\n\t"
       "add.cc.u32       %1, %1, tmp;\n\t"
       "mul.lo.u32       tmp, %5, %9;\n\t"
       "addc.cc.u32      %2, %2, tmp;\n\t"
       "mul.lo.u32       tmp, %5, %10;\n\t"
       "addc.u32         %3, %3, tmp;\n\t"

       "mul.hi.u32       tmp, %5, %8;\n\t"
       "add.cc.u32       %2, %2, tmp;\n\t"
       "mul.hi.u32       tmp, %5, %9;\n\t"
       "addc.u32         %3, %3, tmp;\n\t"

       "mul.lo.u32       tmp, %6, %8;\n\t"
       "add.cc.u32       %2, %2, tmp;\n\t"
       "mul.lo.u32       tmp, %6, %9;\n\t"
       "addc.u32         %3, %3, tmp;\n\t"

       "mad.hi.u32       %3, %6, %8, %3;\n\t"

       "mad.lo.u32       %3, %7, %8, %3;\n\t"
       "}\n\t"
       : "=r"(result.word0()), "=r"(result.word1()), "=r"(result.word2()),
   "=r"(result.word3()) : "r"(a.word0()), "r"(a.word1()), "r"(a.word2()),
   "r"(a.word3()), "r"(b.word0()), "r"(b.word1()), "r"(b.word2()),
   "r"(b.word3())
       );
       return result;
       */

    u64 resultHi, resultLo;
    u64 aHi = a.getPart1(), aLo = a.getPart0();
    u64 bHi = b.getPart1(), bLo = b.getPart0();

    asm("{\n\t"
        ".reg .u64 tmp;\n\t"
        "mul.lo.u64    %0, %2, %4;\n\t"
        "mul.hi.u64    %1, %2, %4;\n\t"
        "mul.lo.u64    tmp, %2, %5;\n\t"
        "add.u64       %1, %1, tmp;\n\t"
        "mul.lo.u64    tmp, %3, %4;\n\t"
        "add.u64       %1, %1, tmp;\n\t"
        "}\n\t"
        : "=l"(resultLo), "=l"(resultHi)
        : "l"(aLo), "l"(aHi), "l"(bLo), "l"(bHi));

    return Int128(resultHi, resultLo);
  }
#endif

  // multiply by a 128 bit value
  HD Int128 operator*(Int128 x) const { return mult128lo(*this, x); }

  HD Int128 operator*(int32_t x) const { return mult128lo(*this, x); }

  HD Int128 operator*(int64_t x) const { return mult128lo(*this, x); }

  HD Int128 operator*(uint32_t x) const { return mult128lo(*this, x); }

  HD Int128 operator*(uint64_t x) const { return mult128lo(*this, x); }

  friend HD Int128 operator*(int32_t l, const Int128& r) { return r * l; }

  friend HD Int128 operator*(int64_t l, const Int128& r) { return r * l; }

  friend HD Int128 operator*(uint32_t l, const Int128& r) { return r * l; }

  friend HD Int128 operator*(uint64_t l, const Int128& r) { return r * l; }
  // breaking this out into a separate function to make it easier to
  // change the implementation
  HD static void divMod(uint64_t dividend, unsigned divisor, uint64_t& quotient,
                        unsigned& remainder) {
    quotient = dividend / divisor;
    remainder = dividend % divisor;
  }

  HD Int128 operator/(int32_t divisor) const {
    Int128 quotient;
    unsigned modulo;
    divMod(divisor, quotient, modulo);
    return quotient;
  }

  HD Int128 operator/(uint32_t divisor) const {
    Int128 quotient;
    unsigned modulo;
    divMod(divisor, quotient, modulo);
    return quotient;
  }

  friend HD Int128 operator/(int32_t l, const Int128& r) {
    return Int128(l) / r.to_uint64_t();
  }

  friend HD Int128 operator/(uint32_t l, const Int128& r) {
    return Int128(l) / r.to_uint64_t();
  }

  HD unsigned operator%(unsigned divisor) {
    Int128 quotient;
    unsigned modulo;
    divMod(divisor, quotient, modulo);
    return modulo;
  }

  HD unsigned operator%(Int128 divisor) {
    return operator%((unsigned) divisor.to_uint64_t());
  }

  /*
    Divide a Int128 by a 32-bit value.
    If each letter is 32 bits:
      ABCD / E
  */
  HD void divMod(unsigned divisor, Int128& quotient, unsigned& modulo) const {
    Int128 thisCopy = *this;  // make a copy, in case 'quotient' is 'this'.
    uint64_t tmp, q;
    unsigned carry = 0;

    tmp = thisCopy.word3();
    divMod(tmp, divisor, q, carry);
    quotient.setWord3(q);

    tmp = ((uint64_t) carry << 32) | thisCopy.word2();
    divMod(tmp, divisor, q, carry);
    quotient.setWord2(q);

    tmp = ((uint64_t) carry << 32) | thisCopy.word1();
    divMod(tmp, divisor, q, carry);
    quotient.setWord1(q);

    tmp = ((uint64_t) carry << 32) | thisCopy.word0();
    divMod(tmp, divisor, q, modulo);
    quotient.setWord0(q);
  }

  HD Int128 operator/(int64_t divisor) const {
    Int128 quotient;
    uint64_t modulo;
    divMod(divisor, quotient, modulo);
    return quotient;
  }

  HD Int128 operator/(uint64_t divisor) const {
    Int128 quotient;
    uint64_t modulo;
    divMod(divisor, quotient, modulo);
    return quotient;
  }

  friend Int128 operator/(int64_t l, const Int128& r) {
    return Int128(l) / r.to_uint64_t();
  }

  friend Int128 operator/(uint64_t l, const Int128& r) {
    return Int128(l) / r.to_uint64_t();
  }

  HD Int128 operator/(const Int128& divisor) const {
    return *this / divisor.to_uint64_t();
  }

  HD Int128 operator%(uint64_t divisor) const {
    Int128 quotient;
    uint64_t modulo;
    divMod(divisor, quotient, modulo);
    return modulo;
  }

  HD void divMod(uint64_t divisor, Int128& quotient, uint64_t& modulo) const {
    if (divisor == 0) {
      modulo = 0xffffffffffffffffull;
      quotient.setPart0(0xffffffffffffffffull);
      quotient.setPart1(0xffffffffffffffffull);
      return;
    }

    if (divisor == 1) {
      modulo = 0;
      quotient = *this;
      return;
    }

    // if the divisor is 32 bits, use the simpler version
    if ((divisor & HI_MASK) == 0) {
      unsigned tmpMod;
      divMod((unsigned) divisor, quotient, tmpMod);
      modulo = tmpMod;
      return;
    }

    divideNewton(divisor, quotient, modulo);
    /*
    Int128 q;
    uint64_t m;
    divModKnuth(divisor, q, m);
    if (quotient != q || m != modulo) {
      printf("divMod error: %016llx%016llx / %llx\n  %016llx%016llx %llx\n
    %016llx%016llx %llx\n", getPart1(), getPart0(), divisor,
    quotient.getPart1(), quotient.getPart0(), modulo, q.getPart1(),
    q.getPart0(), m);
    }
    */
  }

  /*
    Divide a Int128 by a 64-bit value.
    If each letter is 32 bits:
      ABCD / EF
  */
  HD void divModKnuth(uint64_t divisor, Int128& quotient, uint64_t& modulo) {
    // the quotient will be placed here
    unsigned q[3];

    // shift divisor left so its top bit is set and split it into two
    // 32-bit values
    int divisorShift = XMP::countLeadingZeros64(divisor);
    unsigned d[2];
    d[0] = (unsigned) (divisor << divisorShift);
    d[1] = (unsigned) (divisor >> (32 - divisorShift));

    // shift the dividend by the same amount plus 32 bits, and copy it into u[]
    // it is shifted by an extra 32 bits so needsCorrection() can
    // reference u[-1]
    unsigned uData[5], *u = uData + 2;
    shiftAndCopy(uData, divisorShift);

    for (int j = 2; j >= 0; j--, u--) {
      // one word of the quotient
      unsigned qWord;

      // qWord = u[j+1..j+2] / d[0..1]

      if (u[2] == d[1]) {
        qWord = UINT_MAX;
      } else {
        // estimate this word of the quotient by dividing by the top
        // word of the divisor
        qWord = (unsigned) (join64(u + 1) / d[1]);

        // this would be needed if d[] had more than two elements
        // while (needsCorrection(qWord, u, d)) qWord--;
      }

      // remove qWord*d from u
      // if the qWord was too big, the result will be negative, so reduce
      // qWord and add back d until non-negative
      if (removeProduct(u, qWord, d)) {
        do {
          qWord--;
        } while (!addBack(u, d));
      }
      q[j] = qWord;
    }

    quotient.setPart0(join64(q + 0));
    quotient.setPart1(q[2]);

    // modulo is in uData[0..2] shifted by divisorShift
    modulo = (join64(uData) >> divisorShift) |
             ((uint64_t) uData[2] << (64 - divisorShift));

    // test Newton method
    /*
    Int128 altQuotient;
    uint64_t altModulo;
    divideNewton(divisor, altQuotient, altModulo);
    if (altQuotient != quotient || altModulo != modulo) {
      printf("divide fail %016llx %016llx / %016llx\n", part1, getPart0(),
    divisor); assert(altQuotient == quotient); assert(altModulo == modulo);
    }
    */
  }

  // Use Newton-Raphson approximation to compute 2**128 / divisor
  HD static Int128 recipNewton(uint64_t divisor) {
    Int128 inva(0xffffffffffffffffuLL, -divisor);
    float frecip;
    Int128 x;

    int z = XMP::countLeadingZeros64(divisor);
    // 8388607 == 2**23 - 1
    // undershoot a bit (ha) to insure the estimate is low
    // ~2^(23+63-z) = ~2^(86-z) / divisor
    frecip = (8388607.0f * ((uint64_t) 1 << (63 - z))) / divisor;
    // want 2^128 / divisor, have 2^(86-z) / divisor,
    // so shift by 128 - (86-z) = z + 42
    x = Int128(0, (uint64_t) frecip) << (z + 42);

    /*
    // 2**64 * (1-.5**22)
    // removing a small fraction to insure the initial value is low
    frecip = 1.8446739675663041e+19f / divisor;
    x.part1 = (uint64_t)frecip;
    x.part0 = 0;
    */

    // Int128 inva = Int128(0) - divisor;
    // alternates tried, found slower:
    // Int128 inva(0xffffffffffffffffuLL, (~divisor)+1);

    // x = x + Int128::multiply_128_128_hi(x, inva*x);

    // start with reduced-precision operation, becuase only up to 46 bits
    // will be accurate anyway
    x.setPart1(x.getPart1() +
               XMP::multHi64(x.getPart1(), (inva.getPart0() * x.getPart1())));

    x = x + Int128::mult128hi(x, inva * x);
    x = x + Int128::mult128hi(x, inva * x);

    // final check
    uint64_t check = (Int128((uint64_t) 0) - (x * divisor)).getPart0();
    if (check >= divisor)
      x += 1;

    return x;
  }

  /*
    Divide a Int128 by a 64-bit value using Newton-Raphson approximmation.

    1/a =
    x(n+1) = x(n) * (2 - a * x(n))
           = 2*x(n) - a * x(n) * x(n)
           = x(n) + x(n) - a * x(n) * x(n)
           = x(n) + x(n) * (1 - a * x(n))
          ~= x(n) + x(n) * (- a * x(n))
  */
  HD void divideNewton(uint64_t divisor, Int128& quotient,
                       uint64_t& modulo) const {
    Int128 x = recipNewton(divisor);

    quotient = mult128hi(x, *this);
    modulo = (*this - quotient * divisor).getPart0();
    if (modulo >= divisor) {
      modulo -= divisor;
      quotient += 1;
    }
  }

  HD bool isZero() { return getPart1() == 0 && getPart0() == 0; }

  static unsigned rand32() {
    // rand only generates 31 bits
    return (rand() << 16) ^ rand();
  }

  static uint64_t rand64() { return ((uint64_t) rand32() << 32) | rand32(); }

  HD static void clearWords(uint64_t buf[], int len) {
    for (int i = 0; i < len; i++)
      buf[i] = 0;
  }

  HD static void printWords(const char* label, uint64_t buf[], int len) {
#ifndef __CUDA_ARCH__
    std::cout << label << std::hex << std::setfill('0');
    for (int i = len - 1; i >= 0; i--)
      std::cout << " " << std::setw(16) << buf[i];
    std::cout << std::endl;
#endif
  }

 private:
  // shift this left by 'shift' bits (less than 32) and copy the
  // result into 192 bit array 'u' (u[0] is least significant),
  // leaving u[0] empty (makes edge case easier later on)
  //   part1---- getPart0()----     0
  // u[5] u[4] u[3] u[2] u[1] u[0]
  HD void shiftAndCopy(unsigned u[], int shift) {
    u[0] = (unsigned) (getPart0() << shift);
    u[1] = (unsigned) (getPart0() >> (32 - shift));
    u[2] = (unsigned) ((getPart1() << shift) | (getPart0() >> (64 - shift)));
    u[3] = (unsigned) (getPart1() >> (32 - shift));
    u[4] = (unsigned) (getPart1() >> (64 - shift));
  }

  // given a pointer to two 32-bit integers, return them combined
  // into one 64-bit integers with the first one being less significant
  HD uint64_t join64(unsigned* p) { return p[0] | ((uint64_t) p[1] << 32); }

  // removes qWord * (d1<<32 | d0) from u[0..2]
  // returns true iff result is negative
  HD bool removeProduct(unsigned* u, unsigned qWord, unsigned* d) {
    uint64_t pBig = (uint64_t) d[0] * qWord;
    unsigned p = (unsigned) pBig;  // low 32 bits
    unsigned carry, nextCarry;

    carry = (u[0] < p) ? 1 : 0;
    u[0] -= p;
    pBig = (uint64_t) d[1] * qWord + (pBig >> 32);

    p = (unsigned) pBig;

    if (carry)
      nextCarry = u[1] <= p;
    else
      nextCarry = u[1] < p;

    // orig: u[1] - carry < p
    // fail: u[1] = 0, p = 1

    // attempt #2: nextCarry = (u[1] < p + carry) ? 1 : 0;
    // fail: u[1] = 0, p = 0xffffffff,c=1

    u[1] -= p + carry;
    carry = nextCarry;

    p = (unsigned) (pBig >> 32);
    if (carry)
      nextCarry = u[2] <= p;
    else
      nextCarry = u[2] < p;
    u[2] -= p + carry;

    return nextCarry;
  }

  // returns true if qWord is too large:
  //   d[1]*qWord > (((u[1]<<32) + u[0] - qWord*d[0]) << 32) + u[-1]
  HD bool needsCorrection(unsigned qWord, unsigned* u, unsigned* d) {
    // copy qWord into a larger variable just to avoid ugly casts
    uint64_t qWordBig = qWord;

    uint64_t x = join64(u) - qWordBig * d[0];

    // if there are bit set in the high word, then when we shift
    // it up, it will definitely be greater than d[1]*qWord;
    if (x & HI_MASK)
      return true;

    return qWordBig * d[1] > (x << 32) + u[-1];
  }

  // returns 1 on overflow
  HD unsigned addBack(unsigned* u, unsigned* d) {
    // using extra long 'sum' just for the carry bit
    uint64_t sum = (uint64_t) u[0] + d[0];
    u[0] = (unsigned) sum;
    sum = (uint64_t) u[1] + d[1] + (sum >> 32);
    u[1] = (unsigned) sum;
    sum = (uint64_t) u[2] + (sum >> 32);
    u[2] = (unsigned) sum;
    return (unsigned) (sum >> 32);
  }
};

#endif  // RAYJOIN_UTIL_INT128