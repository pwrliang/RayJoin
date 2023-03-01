#ifndef DECIMAL_HXX_
#define DECIMAL_HXX_
#include <type_traits>
namespace aries_acc {

// typedef unsigned char uint8_t;
// typedef int int32_t;

#ifdef LEN_fir
#define NUM_TOTAL_DIG 4
#endif
#ifdef LEN_sec
#define NUM_TOTAL_DIG 8
#endif
#ifdef LEN_thr
#define NUM_TOTAL_DIG 16
#endif
#ifdef LEN_for
#define NUM_TOTAL_DIG 32
#endif
#ifdef LEN_fiv
#define NUM_TOTAL_DIG 64
#endif
#ifdef LEN_six
#define NUM_TOTAL_DIG 128
#endif

#define ARIES_MODE_EMPTY 0  // 0b00
#define MAX_INT32 4294967296
#define MAX_NUM32 (0xFFFFFFFF)

#define ERR_OVER_FLOW 1
#define ERR_DIV_BY_ZERO 2

#define INDEX_LAST_DIG (NUM_TOTAL_DIG - 1)

// precision/scale精度的定义
#define DEFAULT_PRECISION 10  // 默认的precision精度
#define TINYINT_PRECISION 3
#define SMALLINT_PRECISION 5
#define MEDIUMINT_PRECISION 8
#define INT_PRECISION 10
#define BIGINT_PRECISION 19
#define INT128_PRECISION 39
#define DEFAULT_SCALE 0  // 默认的scale精度

#define PER_INT_MAX_BIT 32  // 每个uint32 的最大 bit 数量

#define MAX_BASE10_PER_INT 1000000000

#define PER_DEC_MAX_SCALE 4294967296  // 每个int的值不能大于此值
#define DIV_FIX_INNER_FRAC 6
#define DIV_FIX_EX_FRAC 4  // 计算除法时，正常结果只需要多保留4位小数

#define DIG_PER_INT32 9  // 每个int能够表示的十进制的位数: 9
#define GET_SIGN_FROM_BIT(x) ((x >> 7) & 0x1)  // 对compact decimal，获取符号位
#define SET_SIGN_BIT(x, sign) \
  (x = (x | sign << 7))  // 对compact
                         // decimal，将符号位设置到某一个byte的第一个bit上

#define ARIES_ALIGN(x) __attribute__((aligned(x)))
#define ARIES_PACKED __attribute__((packed))

#ifdef __CUDACC__

#ifndef ARIES_HOST_DEVICE
#define ARIES_HOST_DEVICE __forceinline__ __device__ __host__
#endif

#ifndef ARIES_HOST_DEVICE_NO_INLINE
#define ARIES_HOST_DEVICE_NO_INLINE __device__ __host__
#endif

#ifndef ARIES_DEVICE
#define ARIES_DEVICE __device__
#endif

#ifndef ARIES_LAMBDA
#define ARIES_LAMBDA __device__ __host__
#endif

#ifndef ARIES_DEVICE_FORCE
#define ARIES_DEVICE_FORCE __forceinline__ __device__
#endif

#else  // #ifndef __CUDACC__

#define ARIES_HOST_DEVICE
#define ARIES_HOST_DEVICE_NO_INLINE
#define ARIES_DEVICE
#define ARIES_LAMBDA

#endif  // #ifdef __CUDACC__

typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long uint64_t;
static_assert(sizeof(uint8_t) == 1, "");
static_assert(sizeof(uint16_t) == 2, "");
static_assert(sizeof(uint32_t) == 4, "");
static_assert(sizeof(uint64_t) == 8, "");

typedef char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long int64_t;
static_assert(sizeof(int8_t) == 1, "");
static_assert(sizeof(int16_t) == 2, "");
static_assert(sizeof(int32_t) == 4, "");
static_assert(sizeof(int64_t) == 8, "");

struct ARIES_PACKED CompactDecimal {
  char data[1];
};

struct ARIES_PACKED Decimal {
  uint8_t sign;
  uint8_t prec;
  uint8_t frac;
  uint8_t error;
  uint32_t v[NUM_TOTAL_DIG];

 public:
  ARIES_HOST_DEVICE_NO_INLINE Decimal();
  ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale,
                                      uint32_t m);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale,
                                      const char s[]);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t precision, uint32_t scale,
                                      uint32_t m, const char s[]);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(const CompactDecimal* compact,
                                      uint32_t precision, uint32_t scale,
                                      uint32_t m = ARIES_MODE_EMPTY);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(const char s[]);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(const char s[], int len,
                                      bool base_output);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(const char* s, int len);

  ARIES_HOST_DEVICE_NO_INLINE Decimal(int8_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(int16_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(int32_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(int64_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(uint8_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(uint16_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(uint32_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(uint64_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal(__int128 t);

  ARIES_HOST_DEVICE_NO_INLINE bool ToCompactDecimal(char* buf, int len);
  ARIES_HOST_DEVICE_NO_INLINE char* GetPrecisionScale(char result[]);
  ARIES_HOST_DEVICE_NO_INLINE uint16_t GetError();
  ARIES_HOST_DEVICE_NO_INLINE char* GetDecimal(char result[]);
  ARIES_HOST_DEVICE_NO_INLINE void CheckOverFlow();
  ARIES_HOST_DEVICE_NO_INLINE double GetDouble() const;
  ARIES_HOST_DEVICE_NO_INLINE uint64_t ToUint64() const;

  ARIES_HOST_DEVICE_NO_INLINE void setIntPart(int value, int pos);
  ARIES_HOST_DEVICE_NO_INLINE void setFracPart(int value, int pos);
  ARIES_HOST_DEVICE_NO_INLINE int getIntPart(int pos) const;
  ARIES_HOST_DEVICE_NO_INLINE int getFracPart(int pos) const;
  ARIES_HOST_DEVICE_NO_INLINE Decimal& cast(const Decimal& t);

  ARIES_HOST_DEVICE_NO_INLINE Decimal& truncate(int p);
  ARIES_HOST_DEVICE_NO_INLINE explicit operator bool() const;
  ARIES_HOST_DEVICE_NO_INLINE explicit operator double() const;
  ARIES_HOST_DEVICE_NO_INLINE Decimal operator-() const;

  // signed
  ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=(int8_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=(int16_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=(int32_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=(int64_t t);
  // unsigned
  ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=(uint8_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=(uint16_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=(uint32_t t);
  ARIES_HOST_DEVICE_NO_INLINE Decimal& operator=(uint64_t t);
  // for decimal
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     const Decimal& right);

  // for int8_t
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(int8_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int8_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(int8_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int8_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(int8_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int8_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    int8_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     int8_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    int8_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     int8_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     int8_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     int8_t right);
  // for uint8_t
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint8_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint8_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint8_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint8_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint8_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint8_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    uint8_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     uint8_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    uint8_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     uint8_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     uint8_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     uint8_t right);

  // for int16_t
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(int16_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int16_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(int16_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int16_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(int16_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int16_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    int16_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     int16_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    int16_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     int16_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     int16_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     int16_t right);

  // for uint16_t
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint16_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint16_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint16_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint16_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint16_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint16_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    uint16_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     uint16_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    uint16_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     uint16_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     uint16_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     uint16_t right);

  // for int32_t
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(int32_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int32_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(int32_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int32_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(int32_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int32_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    int32_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     int32_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    int32_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     int32_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     int32_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     int32_t right);

  // for uint32_t
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint32_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint32_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint32_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint32_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint32_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint32_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    uint32_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     uint32_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    uint32_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     uint32_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     uint32_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     uint32_t right);

  // for int64_t
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(int64_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int64_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(int64_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int64_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(int64_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int64_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    int64_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     int64_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    int64_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     int64_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     int64_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     int64_t right);

  // for uint64_t
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint64_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint64_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint64_t left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint64_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint64_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint64_t left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    uint64_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     uint64_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    uint64_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     uint64_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     uint64_t right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     uint64_t right);

  // for float
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(float left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(float left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(float left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(float left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(float left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(float left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    float right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     float right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    float right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     float right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     float right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     float right);

  // for double
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(double left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(double left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(double left,
                                                    const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(double left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(double left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(double left,
                                                     const Decimal& right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                                    double right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                                     double right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                                    double right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                                     double right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                                     double right);
  friend ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                                     double right);

  // for add
  ARIES_DEVICE Decimal& operator+=(const Decimal& d);
  // signed
  ARIES_DEVICE Decimal& operator+=(int8_t i);
  ARIES_DEVICE Decimal& operator+=(int16_t i);
  ARIES_DEVICE Decimal& operator+=(int32_t i);
  ARIES_DEVICE Decimal& operator+=(int64_t i);
  // unsigned
  ARIES_DEVICE Decimal& operator+=(uint8_t i);
  ARIES_DEVICE Decimal& operator+=(uint16_t i);
  ARIES_DEVICE Decimal& operator+=(uint32_t i);
  ARIES_DEVICE Decimal& operator+=(uint64_t i);
  // double / float
  ARIES_DEVICE double operator+=(const float& f);
  ARIES_DEVICE double operator+=(const double& l);
  // self operator
  ARIES_DEVICE Decimal& operator++();
  ARIES_DEVICE Decimal operator++(int32_t);
  // two operators
  friend ARIES_DEVICE Decimal operator+(const Decimal& left,
                                        const Decimal& right);
  // signed
  friend ARIES_DEVICE Decimal operator+(const Decimal& left, int8_t right);
  friend ARIES_DEVICE Decimal operator+(const Decimal& left, int16_t right);
  friend ARIES_DEVICE Decimal operator+(const Decimal& left, int32_t right);
  friend ARIES_DEVICE Decimal operator+(const Decimal& left, int64_t right);
  friend ARIES_DEVICE Decimal operator+(int8_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator+(int16_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator+(int32_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator+(int64_t left, const Decimal& right);
  // unsigned
  friend ARIES_DEVICE Decimal operator+(const Decimal& left, uint8_t right);
  friend ARIES_DEVICE Decimal operator+(const Decimal& left, uint16_t right);
  friend ARIES_DEVICE Decimal operator+(const Decimal& left, uint32_t right);
  friend ARIES_DEVICE Decimal operator+(const Decimal& left, uint64_t right);
  friend ARIES_DEVICE Decimal operator+(uint8_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator+(uint16_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator+(uint32_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator+(uint64_t left, const Decimal& right);
  // double / float
  friend ARIES_DEVICE double operator+(const Decimal& left, float right);
  friend ARIES_DEVICE double operator+(const Decimal& left, double right);
  friend ARIES_DEVICE double operator+(float left, const Decimal& right);
  friend ARIES_DEVICE double operator+(double left, const Decimal& right);

  // for sub
  ARIES_DEVICE Decimal& operator-=(const Decimal& d);
  // signed
  ARIES_DEVICE Decimal& operator-=(int8_t i);
  ARIES_DEVICE Decimal& operator-=(int16_t i);
  ARIES_DEVICE Decimal& operator-=(int32_t i);
  ARIES_DEVICE Decimal& operator-=(int64_t i);
  // unsigned
  ARIES_DEVICE Decimal& operator-=(uint8_t i);
  ARIES_DEVICE Decimal& operator-=(uint16_t i);
  ARIES_DEVICE Decimal& operator-=(uint32_t i);
  ARIES_DEVICE Decimal& operator-=(uint64_t i);
  // double / float
  ARIES_DEVICE double operator-=(const float& f);
  ARIES_DEVICE double operator-=(const double& l);
  // self operator
  ARIES_DEVICE Decimal& operator--();
  ARIES_DEVICE Decimal operator--(int32_t);
  // two operators
  friend ARIES_DEVICE Decimal operator-(const Decimal& left,
                                        const Decimal& right);
  // signed
  friend ARIES_DEVICE Decimal operator-(const Decimal& left, int8_t right);
  friend ARIES_DEVICE Decimal operator-(const Decimal& left, int16_t right);
  friend ARIES_DEVICE Decimal operator-(const Decimal& left, int32_t right);
  friend ARIES_DEVICE Decimal operator-(const Decimal& left, int64_t right);
  friend ARIES_DEVICE Decimal operator-(int8_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator-(int16_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator-(int32_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator-(int64_t left, const Decimal& right);
  // unsigned
  friend ARIES_DEVICE Decimal operator-(const Decimal& left, uint8_t right);
  friend ARIES_DEVICE Decimal operator-(const Decimal& left, uint16_t right);
  friend ARIES_DEVICE Decimal operator-(const Decimal& left, uint32_t right);
  friend ARIES_DEVICE Decimal operator-(const Decimal& left, uint64_t right);
  friend ARIES_DEVICE Decimal operator-(uint8_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator-(uint16_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator-(uint32_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator-(uint64_t left, const Decimal& right);
  // double / float
  friend ARIES_DEVICE double operator-(const Decimal& left, const float right);
  friend ARIES_DEVICE double operator-(const Decimal& left, const double right);
  friend ARIES_DEVICE double operator-(const float left, const Decimal& right);
  friend ARIES_DEVICE double operator-(const double left, const Decimal& right);

  // for multiple
  ARIES_DEVICE Decimal& operator*=(const Decimal& d);
  // signed
  ARIES_DEVICE Decimal& operator*=(int8_t i);
  ARIES_DEVICE Decimal& operator*=(int16_t i);
  ARIES_DEVICE Decimal& operator*=(int32_t i);
  ARIES_DEVICE Decimal& operator*=(int64_t i);
  // unsigned
  ARIES_DEVICE Decimal& operator*=(uint8_t i);
  ARIES_DEVICE Decimal& operator*=(uint16_t i);
  ARIES_DEVICE Decimal& operator*=(uint32_t i);
  ARIES_DEVICE Decimal& operator*=(uint64_t i);
  // double / float
  ARIES_DEVICE double operator*=(const float& f);
  ARIES_DEVICE double operator*=(const double& d);
  // two operators
  friend ARIES_DEVICE Decimal operator*(const Decimal& left,
                                        const Decimal& right);
  // signed
  friend ARIES_DEVICE Decimal operator*(const Decimal& left, int8_t right);
  friend ARIES_DEVICE Decimal operator*(const Decimal& left, int16_t right);
  friend ARIES_DEVICE Decimal operator*(const Decimal& left, int32_t right);
  friend ARIES_DEVICE Decimal operator*(const Decimal& left, int64_t right);
  friend ARIES_DEVICE Decimal operator*(int8_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator*(int16_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator*(int32_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator*(int64_t left, const Decimal& right);
  // unsigned
  friend ARIES_DEVICE Decimal operator*(const Decimal& left, int8_t right);
  friend ARIES_DEVICE Decimal operator*(const Decimal& left, int16_t right);
  friend ARIES_DEVICE Decimal operator*(const Decimal& left, int32_t right);
  friend ARIES_DEVICE Decimal operator*(const Decimal& left, int64_t right);
  friend ARIES_DEVICE Decimal operator*(int8_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator*(int16_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator*(int32_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator*(int64_t left, const Decimal& right);
  // double / float
  friend ARIES_DEVICE double operator*(const Decimal& left, const float right);
  friend ARIES_DEVICE double operator*(const Decimal& left, const double right);
  friend ARIES_DEVICE double operator*(const float left, const Decimal& right);
  friend ARIES_DEVICE double operator*(const double left, const Decimal& right);

  // for division
  ARIES_DEVICE Decimal& DivOrMod(const Decimal& d, bool isMod = false);

  ARIES_DEVICE Decimal& operator/=(const Decimal& d);
  // signed
  ARIES_DEVICE Decimal& operator/=(int8_t i);
  ARIES_DEVICE Decimal& operator/=(int16_t i);
  ARIES_DEVICE Decimal& operator/=(int32_t i);
  ARIES_DEVICE Decimal& operator/=(int64_t i);
  // unsigned
  ARIES_DEVICE Decimal& operator/=(uint8_t i);
  ARIES_DEVICE Decimal& operator/=(uint16_t i);
  ARIES_DEVICE Decimal& operator/=(uint32_t i);
  ARIES_DEVICE Decimal& operator/=(uint64_t i);
  // double / float
  ARIES_DEVICE double operator/=(const float& f);
  ARIES_DEVICE double operator/=(const double& d);
  // two operators
  friend ARIES_DEVICE Decimal operator/(const Decimal& left,
                                        const Decimal& right);
  // signed
  friend ARIES_DEVICE Decimal operator/(const Decimal& left, int8_t right);
  friend ARIES_DEVICE Decimal operator/(const Decimal& left, int16_t right);
  friend ARIES_DEVICE Decimal operator/(const Decimal& left, int32_t right);
  friend ARIES_DEVICE Decimal operator/(const Decimal& left, int64_t right);
  friend ARIES_DEVICE Decimal operator/(int8_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator/(int16_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator/(int32_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator/(int64_t left, const Decimal& right);
  // unsigned
  friend ARIES_DEVICE Decimal operator/(const Decimal& left, uint8_t right);
  friend ARIES_DEVICE Decimal operator/(const Decimal& left, uint16_t right);
  friend ARIES_DEVICE Decimal operator/(const Decimal& left, uint32_t right);
  friend ARIES_DEVICE Decimal operator/(const Decimal& left, uint64_t right);
  friend ARIES_DEVICE Decimal operator/(uint8_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator/(uint16_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator/(uint32_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator/(uint64_t left, const Decimal& right);
  // float
  friend ARIES_DEVICE double operator/(const Decimal& left, const float right);
  friend ARIES_DEVICE double operator/(const Decimal& left, const double right);
  friend ARIES_DEVICE double operator/(const float left, const Decimal& right);
  friend ARIES_DEVICE double operator/(const double left, const Decimal& right);

  // for mod
  ARIES_DEVICE Decimal& operator%=(const Decimal& d);
  // signed
  ARIES_DEVICE Decimal& operator%=(int8_t i);
  ARIES_DEVICE Decimal& operator%=(int16_t i);
  ARIES_DEVICE Decimal& operator%=(int32_t i);
  ARIES_DEVICE Decimal& operator%=(int64_t i);
  // unsigned
  ARIES_DEVICE Decimal& operator%=(uint8_t i);
  ARIES_DEVICE Decimal& operator%=(uint16_t i);
  ARIES_DEVICE Decimal& operator%=(uint32_t i);
  ARIES_DEVICE Decimal& operator%=(uint64_t i);
  // double % float
  ARIES_DEVICE double operator%=(const float& f);
  ARIES_DEVICE double operator%=(const double& d);
  // two operators
  friend ARIES_DEVICE Decimal operator%(const Decimal& left,
                                        const Decimal& right);
  // signed
  friend ARIES_DEVICE Decimal operator%(const Decimal& left, int8_t right);
  friend ARIES_DEVICE Decimal operator%(const Decimal& left, int16_t right);
  friend ARIES_DEVICE Decimal operator%(const Decimal& left, int32_t right);
  friend ARIES_DEVICE Decimal operator%(const Decimal& left, int64_t right);
  friend ARIES_DEVICE Decimal operator%(int8_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator%(int16_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator%(int32_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator%(int64_t left, const Decimal& right);
  // unsigned
  friend ARIES_DEVICE Decimal operator%(const Decimal& left, uint8_t right);
  friend ARIES_DEVICE Decimal operator%(const Decimal& left, uint16_t right);
  friend ARIES_DEVICE Decimal operator%(const Decimal& left, uint32_t right);
  friend ARIES_DEVICE Decimal operator%(const Decimal& left, uint64_t right);
  friend ARIES_DEVICE Decimal operator%(uint8_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator%(uint16_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator%(uint32_t left, const Decimal& right);
  friend ARIES_DEVICE Decimal operator%(uint64_t left, const Decimal& right);
  // float
  friend ARIES_DEVICE double operator%(const Decimal& left, const float right);
  friend ARIES_DEVICE double operator%(const Decimal& left, const double right);
  friend ARIES_DEVICE double operator%(const float left, const Decimal& right);
  friend ARIES_DEVICE double operator%(const double left, const Decimal& right);

  ARIES_HOST_DEVICE_NO_INLINE void AddDecimalOnCpu(const Decimal& d);
  ARIES_HOST_DEVICE_NO_INLINE void MulDecimalOnCpu(const Decimal& d);

 public:
  // 仅计算结果精度,可针对各精度数据四则混合运算后的结果精度进行计算
  ARIES_HOST_DEVICE_NO_INLINE void CalcAddTargetPrecision(const Decimal& d);
  ARIES_HOST_DEVICE_NO_INLINE void CalcSubTargetPrecision(const Decimal& d);
  ARIES_HOST_DEVICE_NO_INLINE void CalcMulTargetPrecision(const Decimal& d);
  ARIES_HOST_DEVICE_NO_INLINE void CalcDivTargetPrecision(const Decimal& d);
  ARIES_HOST_DEVICE_NO_INLINE void CalcModTargetPrecision(const Decimal& d);
  /* CalcTruncTargetPrecision
   * int p: > 0 try to truncate frac part to p scale
   *        = 0 try to truncate to integer
   *        < 0 try to truncate to integer, and intg part will be truncated
   * */
  ARIES_HOST_DEVICE_NO_INLINE bool isZero() const;

 public:
  // 检查和设置实际精度
  ARIES_HOST_DEVICE_NO_INLINE void CheckAndSetRealPrecision();

 protected:
  // decimal按10进制进行右移n位，相当除以 10的n次方
  ARIES_HOST_DEVICE_NO_INLINE Decimal& operator>>(int n);
  // decimal按10进制进行左移n位，相当乘以 10的n次方
  ARIES_HOST_DEVICE_NO_INLINE Decimal& operator<<(int n);
  // 将decimal变成一个整数decimal
  ARIES_HOST_DEVICE_NO_INLINE Decimal& GenIntDecimal(int shift = 0);
  // 将两个整数decimal折半取值
  ARIES_DEVICE Decimal HalfIntDecimal(const Decimal d1, const Decimal d2);
  ARIES_DEVICE Decimal DivInt(const Decimal ds, const Decimal dt,
                              Decimal& residuel, uint32_t dtHitBit);

  ARIES_HOST_DEVICE_NO_INLINE int32_t GetPowers10(int i) const;

  // 根据精度获取其最大值
  ARIES_HOST_DEVICE_NO_INLINE void GenMaxDecByPrecision();
  // 对齐加减法的两个操作数的小数位
  ARIES_HOST_DEVICE_NO_INLINE void AlignAddSubData(Decimal& d);
  ARIES_HOST_DEVICE_NO_INLINE void initialize(uint32_t pr, uint32_t fc,
                                              uint32_t m);
  ARIES_HOST_DEVICE_NO_INLINE void CopyValue(Decimal& d);
  ARIES_HOST_DEVICE_NO_INLINE bool StringToDecimal(char* str);
  ARIES_HOST_DEVICE_NO_INLINE bool StringToDecimal(char* str, int len);
  // 以int进行除法处理
  ARIES_DEVICE Decimal& DivByInt(const Decimal& d, int shift,
                                 bool isMod = false);
  // 以long进行除法处理
  ARIES_DEVICE Decimal& DivByInt64(const Decimal& d, int shift,
                                   bool isMod = false);
};

ARIES_HOST_DEVICE_NO_INLINE Decimal abs(Decimal decimal);
// 根据精度信息获得compact需要存储的 bytes
ARIES_HOST_DEVICE_NO_INLINE int GetDecimalRealBytes(uint16_t precision,
                                                    uint16_t scale);
}  // namespace aries_acc

namespace std {
template <>
struct is_integral<aries_acc::Decimal> : public true_type {};

}  // namespace std
#endif
