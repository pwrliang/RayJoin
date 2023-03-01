#include <cassert>

#include "decimal.h"

extern "C" __device__ void __direct_callable__dummy_func() {}

namespace aries_acc {

#define aries_is_digit(c) ((c) >= '0' && (c) <= '9')
#define aries_max(a, b) (((a) > (b)) ? (a) : (b))
#define aries_min(a, b) (((a) > (b)) ? (b) : (a))
#define aries_abs(a) (((a) < (0)) ? (-a) : (a))
#define aries_is_digit(c) ((c) >= '0' && (c) <= '9')

__device__ __managed__ uint32_t __ARRAY_SCALE[] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5};

__device__ __managed__ uint32_t __POW10_ARRAY[][NUM_TOTAL_DIG] = {
    {0x0000000a},
    {0x00000064},
    {0x000003e8},
    {0x00002710},
    {0x000186a0},
    {0x000f4240},
    {0x00989680},
    {0x05f5e100},
    {0x3b9aca00},
    {0x540be400, 0x00000002},
    {0x4876e800, 0x00000017},
    {0xd4a51000, 0x000000e8},
    {0x4e72a000, 0x00000918},
    {0x107a4000, 0x00005af3},
    {0xa4c68000, 0x00038d7e},
    {0x6fc10000, 0x002386f2},
    {0x5d8a0000, 0x01634578},
    {0xa7640000, 0x0de0b6b3},
    {0x89e80000, 0x8ac72304},
    {0x63100000, 0x6bc75e2d, 0x00000005},
    {0xdea00000, 0x35c9adc5, 0x00000036},
    {0xb2400000, 0x19e0c9ba, 0x0000021e},
    {0xf6800000, 0x02c7e14a, 0x0000152d},
    {0xa1000000, 0x1bcecced, 0x0000d3c2},
    {0x4a000000, 0x16140148, 0x00084595},
    {0xe4000000, 0xdcc80cd2, 0x0052b7d2},
    {0xe8000000, 0x9fd0803c, 0x033b2e3c},
    {0x10000000, 0x3e250261, 0x204fce5e},
    {0xa0000000, 0x6d7217ca, 0x431e0fae, 0x00000001},
    {0x40000000, 0x4674edea, 0x9f2c9cd0, 0x0000000c},
    {0x80000000, 0xc0914b26, 0x37be2022, 0x0000007e},
    {0x00000000, 0x85acef81, 0x2d6d415b, 0x000004ee},
    {0x00000000, 0x38c15b0a, 0xc6448d93, 0x0000314d},
    {0x00000000, 0x378d8e64, 0xbead87c0, 0x0001ed09},
    {0x00000000, 0x2b878fe8, 0x72c74d82, 0x00134261},
    {0x00000000, 0xb34b9f10, 0x7bc90715, 0x00c097ce},
    {0x00000000, 0x00f436a0, 0xd5da46d9, 0x0785ee10},
    {0x00000000, 0x098a2240, 0x5a86c47a, 0x4b3b4ca8},
    // {0x00000000,0x5f655680,0x8943acc4,0xf050fe93,0x00000002},
    // {0x00000000,0xb9f56100,0x5ca4bfab,0x6329f1c3,0x0000001d},
    // {0x00000000,0x4395ca00,0x9e6f7cb5,0xdfa371a1,0x00000125},
    // {0x00000000,0xa3d9e400,0x305adf14,0xbc627050,0x00000b7a},
    // {0x00000000,0x6682e800,0xe38cb6ce,0x5bd86321,0x000072cb},
    // {0x00000000,0x011d1000,0xe37f2410,0x9673df52,0x00047bf1},
    // {0x00000000,0x0b22a000,0xe2f768a0,0xe086b93c,0x002cd76f},
    // {0x00000000,0x6f5a4000,0xddaa1640,0xc5433c60,0x01c06a5e},
    // {0x00000000,0x59868000,0xa8a4de84,0xb4a05bc8,0x118427b3},
    // {0x00000000,0x7f410000,0x9670b12b,0x0e4395d6,0xaf298d05}
};

ARIES_HOST_DEVICE_NO_INLINE int aries_is_space(int ch) {
  return (unsigned long) (ch - 9) < 5u || ' ' == ch;
}

ARIES_HOST_DEVICE_NO_INLINE int aries_atoi(const char* str, const char* end) {
  int sign;
  int n = 0;
  const char* p = str;

  while (p != end && aries_is_space(*p))
    p++;
  if (p != end) {
    sign = ('-' == *p) ? -1 : 1;
    if ('+' == *p || '-' == *p)
      p++;

    for (n = 0; p != end && aries_is_digit(*p); p++)
      n = 10 * n + (*p - '0');

    if (sign == -1)
      n = -n;
  }
  return n;
}

ARIES_HOST_DEVICE_NO_INLINE int aries_atoi(const char* str) {
  int sign;
  int n = 0;
  const char* p = str;

  while (aries_is_space(*p))
    p++;

  sign = ('-' == *p) ? -1 : 1;
  if ('+' == *p || '-' == *p)
    p++;

  for (n = 0; aries_is_digit(*p); p++)
    n = 10 * n + (*p - '0');

  if (sign == -1)
    n = -n;
  return n;
}

ARIES_HOST_DEVICE_NO_INLINE int64_t aries_atol(const char* str,
                                               const char* end) {
  int sign;
  int64_t n = 0;
  const char* p = str;

  while (p != end && aries_is_space(*p))
    p++;
  if (p != end) {
    sign = ('-' == *p) ? -1 : 1;
    if ('+' == *p || '-' == *p)
      p++;

    for (n = 0; p != end && aries_is_digit(*p); p++)
      n = 10 * n + (*p - '0');

    if (sign == -1)
      n = -n;
  }
  return n;
}

ARIES_HOST_DEVICE_NO_INLINE int64_t aries_atol(const char* str) {
  int sign;
  int64_t n = 0;
  const char* p = str;

  while (aries_is_space(*p))
    p++;

  sign = ('-' == *p) ? -1 : 1;
  if ('+' == *p || '-' == *p)
    p++;

  for (n = 0; aries_is_digit(*p); p++)
    n = 10 * n + (*p - '0');

  if (sign == -1)
    n = -n;
  return n;
}

ARIES_HOST_DEVICE_NO_INLINE int aries_strlen(const char* str) {
  const char* p = str;
  while (*p++)
    ;

  return (int) (p - str - 1);
}

ARIES_HOST_DEVICE_NO_INLINE char* aries_strcpy(char* strDest,
                                               const char* strSrc) {
  if (strDest == strSrc) {
    return strDest;
  }
  assert((strDest != NULL) && (strSrc != NULL));
  char* address = strDest;
  while ((*strDest++ = *strSrc++))
    ;
  return address;
}

ARIES_HOST_DEVICE_NO_INLINE char* aries_strncpy(char* strDest,
                                                const char* strSrc,
                                                unsigned int count) {
  if (strDest == strSrc) {
    return strDest;
  }
  assert((strDest != NULL) && (strSrc != NULL));
  char* address = strDest;
  while (count-- && *strSrc)
    *strDest++ = *strSrc++;
  *strDest = 0;
  return address;
}

ARIES_HOST_DEVICE_NO_INLINE char* aries_strcat(char* strDes,
                                               const char* strSrc) {
  assert((strDes != NULL) && (strSrc != NULL));
  char* address = strDes;
  while (*strDes)
    ++strDes;
  while ((*strDes++ = *strSrc++))
    ;
  return address;
}

ARIES_HOST_DEVICE_NO_INLINE char* aries_strncat(char* strDes,
                                                const char* strSrc,
                                                unsigned int count) {
  assert((strDes != NULL) && (strSrc != NULL));
  char* address = strDes;
  while (*strDes)
    ++strDes;
  while (count-- && *strSrc)
    *strDes++ = *strSrc++;
  *strDes = 0;
  return address;
}

ARIES_HOST_DEVICE_NO_INLINE char* aries_strchr(const char* str, int ch) {
  while (*str && *str != (char) ch)
    str++;

  if (*str == (char) ch) {
    return ((char*) str);
  }
  return 0;
}

ARIES_HOST_DEVICE_NO_INLINE char* aries_sprintf(char* dst, const char* fmt,
                                                int v) {
  int startPos = 0;
  int len = aries_strlen(fmt);
  // only support format : %d, %010d
  if (fmt[startPos++] != '%' || fmt[len - 1] != 'd') {
    assert(0);
    return dst;
  }

  int outLen = -1;
  bool fillwithz = false;
  if (fmt[startPos] == '0') {
    fillwithz = true;
    ++startPos;
  }
  char tmp[128];
  if (startPos + 1 < len) {
    aries_strncpy(tmp, fmt + startPos, len - startPos - 1);
    outLen = aries_atoi(tmp);
  }
  // no out
  if (outLen == 0) {
    dst[0] = '0';
    dst[1] = 0;
    return dst;
  }
  int negsign = 0;
  int val = v;
  startPos = 0;
  if (val < 0) {
    negsign = 1;
    val = -val;
  }
  do {
    tmp[startPos++] = char('0' + val % 10);
    val /= 10;
  } while (val > 0);

  len = startPos;
  startPos = 0;
  if (negsign) {
    dst[startPos++] = '-';
  }
  if (outLen == -1) {
    if (len == 0) {
      dst[startPos++] = '0';
    } else {
      for (int i = len - 1; i >= 0; i--) {
        dst[startPos++] = tmp[i];
      }
    }
    dst[startPos] = 0;
  } else {
    int realLen = len + negsign;
    if (fillwithz) {
      int rep0 = outLen - realLen;
      if (rep0 > 0) {
        for (int i = 0; i < rep0; i++) {
          dst[startPos++] = '0';
        }
      }
    }
    int cpylen = outLen - startPos;
    cpylen = cpylen > len ? len : cpylen;
    for (int i = cpylen - 1; i >= 0; i--) {
      dst[startPos++] = tmp[i];
    }
    dst[startPos] = 0;
  }
  return dst;
}

ARIES_HOST_DEVICE_NO_INLINE void* aries_memset(void* dst, int val,
                                               unsigned long ulcount) {
  if (!dst)
    return 0;
  char* pchdst = (char*) dst;
  while (ulcount--)
    *pchdst++ = (char) val;

  return dst;
}

ARIES_HOST_DEVICE_NO_INLINE void* aries_memcpy(void* dst, const void* src,
                                               unsigned long ulcount) {
  if (!(dst && src))
    return 0;
  if (!ulcount)
    return dst;
  char* pchdst = (char*) dst;
  char* pchsrc = (char*) src;
  while (ulcount--)
    *pchdst++ = *pchsrc++;

  return dst;
}

ARIES_HOST_DEVICE_NO_INLINE int aries_strcmp(const char* source,
                                             const char* dest) {
  int ret = 0;
  if (!source || !dest)
    return -2;
  while (!(ret = *(unsigned char*) source - *(unsigned char*) dest) && *dest) {
    source++;
    dest++;
  }

  if (ret < 0)
    ret = -1;
  else if (ret > 0)
    ret = 1;

  return (ret);
}

ARIES_HOST_DEVICE_NO_INLINE char* aries_strstr(const char* strSrc,
                                               const char* str) {
  assert(strSrc != NULL && str != NULL);
  const char* s = strSrc;
  const char* t = str;
  for (; *strSrc; ++strSrc) {
    for (s = strSrc, t = str; *t && *s == *t; ++s, ++t)
      ;
    if (!*t)
      return (char*) strSrc;
  }
  return 0;
}

ARIES_HOST_DEVICE_NO_INLINE char* aries_strerase(char* strSrc, int n) {
  char* strDes = (char*) malloc((aries_strlen(strSrc) - 1) * sizeof(char));
  char* address = strDes;
  for (int i = 0; i < n; i++) {
    *strDes++ = *strSrc++;
  }
  strSrc++;
  while ((*strDes++ = *strSrc++))
    ;
  return address;
}

ARIES_HOST_DEVICE_NO_INLINE int32_t abs_cmp(int32_t* a, const int32_t* b) {
  int32_t res = 0;
  // #pragma unroll
  for (int i = NUM_TOTAL_DIG - 1; i >= 0 && res == 0; i--) {
    res = a[i] - b[i];
  }
  return res;
}

ARIES_DEVICE void abs_lshift(uint32_t* a, int len, int n, uint32_t* res) {
  memset(res, 0x00, sizeof(uint32_t) * NUM_TOTAL_DIG);
  uint32_t carry = 0;
  uint64_t temp;
  // #pragma unroll
  for (int i = 0; i < len; i++) {
    carry = 0;
    // #pragma unroll
    for (int j = 0; j < __ARRAY_SCALE[n - 1]; j++) {
      if (i + j > len) {
        break;
      }
      temp = (uint64_t) a[i] * (uint64_t) __POW10_ARRAY[n - 1][j] + res[i + j] +
             carry;
      carry = temp >> 32;
      res[i + j] = temp & 0x00000000ffffffff;
    }
    if (i + __ARRAY_SCALE[n - 1] < len) {
      res[i + __ARRAY_SCALE[n - 1]] = carry;
    }
  }
}

ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal()
    : Decimal(DEFAULT_PRECISION, DEFAULT_SCALE) {}

ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale)
    : Decimal(precision, scale, (uint32_t) ARIES_MODE_EMPTY) {}

// 构造函数,传入精度信息
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale,
                                             uint32_t m) {
  // 初始化精度信息
  initialize(precision, scale, 0);
}

// 构造函数，传入精度和字符串
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale,
                                             const char s[])
    : Decimal(precision, scale, ARIES_MODE_EMPTY, s) {}

// 根据精度和字符串构造相应decimal
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t precision, uint32_t scale,
                                             uint32_t m, const char s[]) {
  // 对 this 初始化精度信息
  initialize(precision, scale, 0);
  // 根据字符串 s 所提供的信息，构造 Decimal d，此时 d 的精度信息为初始精度信息
  Decimal d(s);
  // 将 d 的内容根据对 this 精度信息要求的精度信息做出 截断操作，并返回给 this
  cast(d);
}

// CompactDecimal to Decimal，根据 compact 和 精度信息完成构造
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const CompactDecimal* compact,
                                             uint32_t precision, uint32_t scale,
                                             uint32_t m) {
  initialize(precision, scale, 0);
  int len = GetDecimalRealBytes(precision, scale);
  aries_memcpy((char*) (v), compact->data, len);
  char* temp = ((char*) (v));
  temp += len - 1;
  sign = GET_SIGN_FROM_BIT(*temp);
  *temp = *temp & 0x7f;
}

// 构造函数，传入字符串s
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const char s[]) {
  // 初始化精度信息
  initialize(0, 0, 0);
  // 将字符串转化为Decimal
  bool success = StringToDecimal((char*) s);
}

// 构造函数，根据传入字符串s和字符串长度构造decimal
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const char* s, int len) {
  // 初始化精度信息
  initialize(0, 0, 0);
  // 将字符串转化为Decimal
  bool success = StringToDecimal((char*) s);
}

// 通过 16进制的输出格式 即GetDecimal 的得到的字符串 构造 Decimal
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(const char s[], int len,
                                             bool base_output) {
  // 初始化精度信息
  initialize(0, 0, 0);
  // 第一个字符表示符号
  if (s[0] == '+')
    sign = 0;
  else
    sign = 1;
  // 后五个字符表示frac (xxx)
  int t = len - 2;
  for (int i = t - 2; i <= t; i++) {
    frac = frac * 10 + (s[i] - '0');
  }
  t = len - 5;
  // 之后每8个char代表一个数 len中1个char表示sign 5个char表示frac
  for (int i = 0; i < (len - 6) / 8; i++) {
    t -= 8;
    for (int j = 0; j < 8; j++) {
      v[i] = v[i] * 16;
      if (s[t + j] <= '9') {
        v[i] += s[t + j] - '0';
      } else if (s[t + j] >= 'A') {
        v[i] += s[t + j] - 'A' + 10;
      } else {
        error = 1;
      }
    }
  }
  // 这里向上估算 一个uint32十进制下位数为10
  prec = ((len - 6) / 8) * 10;
}

// 构造函数， int8_t
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int8_t t) {
  initialize(TINYINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
  v[0] = t;
  if (t < 0) {
    sign = 1;
    v[0] = -v[0];
  }
}

// 构造函数， int16_t
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int16_t t) {
  initialize(SMALLINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
  v[0] = t;
  if (t < 0) {
    sign = 1;
    v[0] = -v[0];
  }
}

// 构造函数， int32_t
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int32_t t) {
  initialize(INT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
  if (t < 0) {
    sign = 1;
    t = -t;
  }
  v[0] = t % PER_DEC_MAX_SCALE;
}

// 构造函数， int64_t
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(int64_t t) {
  initialize(BIGINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
  if (t < 0) {
    sign = 1;
    t = -t;
  }
  v[1] = t / PER_DEC_MAX_SCALE;
  v[0] = t % PER_DEC_MAX_SCALE;
}

ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal( __int128 t) {
  initialize(INT128_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
  if (t < 0) {
    sign = 1;
    t = -t;
  }

  aries_memcpy(v, &t, sizeof(__int128));
}

// 构造函数
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint8_t t) {
  initialize(TINYINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
  v[0] = t;
}

// 构造函数
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint16_t t) {
  initialize(SMALLINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
  v[0] = t;
}

// 构造函数
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint32_t t) {
  initialize(INT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
  v[0] = t % PER_DEC_MAX_SCALE;
}

// 构造函数
ARIES_HOST_DEVICE_NO_INLINE Decimal::Decimal(uint64_t t) {
  initialize(BIGINT_PRECISION, DEFAULT_SCALE, ARIES_MODE_EMPTY);
  v[1] = t / PER_DEC_MAX_SCALE;
  v[0] = t % PER_DEC_MAX_SCALE;
}

// Decimal To CompactDecimal
ARIES_HOST_DEVICE_NO_INLINE bool Decimal::ToCompactDecimal(char* buf, int len) {
  aries_memcpy(buf, (char*) (v), len);
  SET_SIGN_BIT(buf[len - 1], sign);
  return true;
}

// 获取精度信息，将prec和frac打包成字符串返回
ARIES_HOST_DEVICE_NO_INLINE char* Decimal::GetPrecisionScale(char result[]) {
  char temp[8];
  int prec0 = prec;
  int frac0 = frac;
  aries_sprintf(temp, "%d", prec0);
  aries_strcpy(result, temp);
  aries_strcat(result, ",");
  aries_sprintf((char*) temp, "%d", frac0);
  aries_strcat(result, temp);
  return result;
}

// 返回错误代码
ARIES_HOST_DEVICE_NO_INLINE uint16_t Decimal::GetError() { return error; }

// 将 decimal 放入字符串 result 中输出 以 16 进制输出
ARIES_HOST_DEVICE_NO_INLINE char* Decimal::GetDecimal(char result[]) {
  char numberDict[] = {'0', '1', '2', '3', '4', '5', '6', '7',
                       '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
  int index = 0;
  if (sign == 0) {
    result[index] = '+';
  } else {
    result[index] = '-';
  }
  int flag = 0;
  uint32_t tmp = 0;
  // #pragma unroll
  for (int i = INDEX_LAST_DIG; i >= 0; i--) {
    if (v[i] != 0) {
      flag = 1;
    }
    if (flag == 1) {
      tmp = v[i];
      index += 8;
      // #pragma unroll
      for (int j = 0; j < 8; j++) {
        result[index - j] = numberDict[tmp & 0xF];
        tmp >>= 4;
      }
    }
  }
  index++;
  result[index++] = '(';
  result[index++] = frac / 100 + '0';
  result[index++] = frac / 10 % 10 + '0';
  result[index++] = frac % 10 + '0';
  result[index++] = ')';
  result[index] = '\0';
  return result;
}

// 检查溢出 用 double 粗略检查
ARIES_HOST_DEVICE_NO_INLINE void Decimal::CheckOverFlow() {
  int i = INDEX_LAST_DIG;
  for (; i >= 0; i--) {
    if (v[i] != 0) {
      break;
    }
  }
  int prec0 = i * DIG_PER_INT32;
  double maxUint32 = 4.294967295;
  double tmpMul = 1;
  for (int j = 0; j < i; j++) {
    tmpMul *= maxUint32;
  }
  tmpMul *= v[i];
  int tt = (int) tmpMul;
  while (tt > 1) {
    tt /= 10;
    prec0++;
  }
  // 声明位数小于实际位数
  if (prec < prec0) {
    error = ERR_OVER_FLOW;
  }
}

/*
 * integer/frac part by pos index
 *   0: value of 0 int
 *   1: value of 1 int
 *   2: value of 2 int
 *   3: value of 3 int
 * */

// rateup 中 时间表示才会用到的函数
ARIES_HOST_DEVICE_NO_INLINE void Decimal::setIntPart(int value, int pos) {
  v[0] += value * GetPowers10(frac) % PER_DEC_MAX_SCALE;
  v[1] += value * GetPowers10(frac) / PER_DEC_MAX_SCALE;
}

// rateup 中 时间表示才会用到的函数
ARIES_HOST_DEVICE_NO_INLINE void Decimal::setFracPart(int value, int pos) {
  v[pos] += value;
}

// rateup 中 时间表示才会用到的函数
ARIES_HOST_DEVICE_NO_INLINE int Decimal::getIntPart(int pos) const {
  return v[1] * PER_DEC_MAX_SCALE + v[0] / GetPowers10(frac);
}

// rateup 中 时间表示才会用到的函数
ARIES_HOST_DEVICE_NO_INLINE int Decimal::getFracPart(int pos) const {
  return v[0] % GetPowers10(frac);
}

// global method
ARIES_HOST_DEVICE_NO_INLINE Decimal abs(Decimal decimal) {
  decimal.sign = 0;
  return decimal;
}

// 获取 Decimal To CompactDecimal 需要的 Byte
ARIES_HOST_DEVICE_NO_INLINE int GetDecimalRealBytes(uint16_t precision,
                                                    uint16_t scale) {
  int needBytes = precision / DIG_PER_INT32 * 4;
  switch (precision % DIG_PER_INT32) {
  case 0:
    needBytes += 0;
    break;
  case 1:
    needBytes += 1;  // 4个bit < 1 个字节
    break;
  case 2:
    needBytes += 1;  // 7个bit < 1 个字节
    break;
  case 3:
    needBytes += 2;  // 10个bit < 2 个字节
    break;
  case 4:
    needBytes += 2;  // 14个bit < 2 个字节
    break;
  case 5:
    needBytes += 3;  // 17个bit <  3个字节
    break;
  case 6:
    needBytes += 3;  // 20个bit < 3 个字节
    break;
  case 7:
    needBytes += 4;  // 24个bit < 4 个字节
    break;
  case 8:
    needBytes += 4;  // 27个bit < 4 个字节
    break;
  }
  return needBytes;
}

// 截断函数，根据 精度信息 prec，frac 对 t.v 进行截断
ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::cast(const Decimal& t) {
  sign = t.sign;
  // 小数部分可能需要扩充
  if (frac >= t.frac) {
    // 查看小数部分
    int shift = frac - t.frac;
    // 需要扩充
    if (shift != 0) {
      aries_memcpy(v, t.v, sizeof(t.v));
      *this << shift;
    }
    // 不需要扩充
    else {
      aries_memcpy(v, t.v, sizeof(t.v));
    }
    // 查看整数部分
    // 如果规定的规格下数值的整数部分大小 小于
    // 其原本应该占据的部分大小,那么需要查看是否溢出
    if ((prec - frac) < (t.prec - t.frac)) {
      // 需要检查正数部分是否溢出 CheckOverFlow()
      CheckOverFlow();
    }
  }
  //
  else {
    // 小数部分需要缩减
    // 小数需要右移,断尾
    // 向右要缩进了几个 int 单位
    int shift = t.frac - frac;
    aries_memcpy(v, t.v, sizeof(t.v));
    // TODO 是否需要 四舍五入  如果不需要 这里可使用 *this >> shift;
    *this >> shift;
    // 如需要四舍五入
    // uint64_t temp = 0;
    // uint32_t remainder = 0;
    // while ( shift>DIG_PER_INT32){
    //     for (int i = INDEX_LAST_DIG; i>=0 ; i--){
    //         temp = remainder * PER_DEC_MAX_SCALE + v[i];
    //         v[i] = temp % PER_DEC_MAX_SCALE;
    //         remainder = temp / PER_DEC_MAX_SCALE;
    //     }
    //     shift -= DIG_PER_INT32;
    // }
    // uint32_t pow10n = GetPowers10(shift);
    // int jw = 0;
    // if( v[0] % pow10n / (pow10n/10) >= 5){
    // 	jw = 1;
    // }
    // remainder = 0;
    // for (int i = INDEX_LAST_DIG; i>=0 ; i--){
    //     temp = remainder * PER_DEC_MAX_SCALE + v[i];
    //     v[i] = temp % PER_DEC_MAX_SCALE;
    //     remainder = temp / PER_DEC_MAX_SCALE;
    // }
    // if( jw == 1 ){
    //     asm  volatile ("add.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(v[0]),
    //     "r"(1)); // #pragma unroll for(int32_t i = 1; i < NUM_TOTAL_DIG; i++)
    //         asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(v[i]) : "r"(v[i]),
    //         "r"(0));
    // }
    CheckOverFlow();
  }
  return *this;
}

/* CalcTruncTargetPrecision
 * int p: > 0 try to truncate frac part to p scale
 *        = 0 try to truncate to integer
 *        < 0 try to truncate to integer, and intg part will be truncated
 * */
// 直接截断函数
ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::truncate(int p) {
  if (p > 0) {
    if (frac >= p) {
      *this >> frac - p;
      prec -= frac - p;
    } else {
      *this << p - frac;
      prec += p - frac;
    }
    frac = p;
  } else {
    p = -p;
    // 让 frac = 0 的同时，该数字末尾是 abs(p) 个 0
    int intg = prec - frac;
    if (intg > p) {
      *this >> frac + p;
      *this << p;
      prec = intg;
    } else {
      aries_memset(v, 0x00, sizeof(v));
      prec = 1;
    }
    frac = 0;
  }

  return *this;
}

ARIES_HOST_DEVICE_NO_INLINE Decimal::operator bool() const { return !isZero(); }

ARIES_HOST_DEVICE_NO_INLINE Decimal::operator double() const {
  return GetDouble();
}

ARIES_HOST_DEVICE_NO_INLINE Decimal Decimal::operator-() const {
  Decimal tmp = *this;
  if (tmp.sign == 0) {
    tmp.sign = 1;
  } else {
    tmp.sign = 0;
  }
  return tmp;
}

// signed
ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::operator=(int8_t t) {
  Decimal tmp(t);
  *this = tmp;
  return *this;
}

ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::operator=(int16_t t) {
  Decimal tmp(t);
  *this = tmp;
  return *this;
}

ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::operator=(int32_t t) {
  Decimal tmp(t);
  *this = tmp;
  return *this;
}

ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::operator=(int64_t t) {
  Decimal tmp(t);
  *this = tmp;
  return *this;
}

// unsigned
ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::operator=(uint8_t t) {
  Decimal tmp(t);
  *this = tmp;
  return *this;
}

ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::operator=(uint16_t t) {
  Decimal tmp(t);
  *this = tmp;
  return *this;
}

ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::operator=(uint32_t t) {
  Decimal tmp(t);
  *this = tmp;
  return *this;
}

ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::operator=(uint64_t t) {
  Decimal tmp(t);
  *this = tmp;
  return *this;
}

// for decimal
ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                           const Decimal& right) {
  long long temp = 0;
  if (left.sign != right.sign) {
    // 符号不同
    if (left.sign == 0) {
      return true;
    }
    return false;
  } else {
    // 符号相同
    Decimal l(left);
    Decimal r(right);
    if (l.frac != r.frac) {
      l.AlignAddSubData(r);
    }
    if (left.sign == 0) {
      // #pragma unroll
      for (int i = INDEX_LAST_DIG; i >= 0; i--) {
        if (temp = (long long) l.v[i] - r.v[i]) {
          return temp > 0;
        }
      }
    } else {
      // #pragma unroll
      for (int i = INDEX_LAST_DIG; i >= 0; i--) {
        if (temp = (long long) l.v[i] - r.v[i]) {
          return temp < 0;
        }
      }
    }
  }
  return false;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                            const Decimal& right) {
  return !(left < right);
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                           const Decimal& right) {
  long long temp;
  if (left.sign != right.sign) {
    // 符号不同
    if (left.sign == 0) {
      return false;
    }
    return true;
  } else {
    // 符号相同
    Decimal l(left);
    Decimal r(right);
    if (l.frac != r.frac) {
      l.AlignAddSubData(r);
    }
    if (left.sign == 0) {
      // #pragma unroll
      for (int i = INDEX_LAST_DIG; i >= 0; i--) {
        if (temp = (long long) l.v[i] - r.v[i]) {
          return temp < 0;
        }
      }
    } else {
      // #pragma unroll
      for (int i = INDEX_LAST_DIG; i >= 0; i--) {
        if (temp = (long long) l.v[i] - r.v[i]) {
          return temp > 0;
        }
      }
    }
  }
  return false;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                            const Decimal& right) {
  return !(left > right);
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                            const Decimal& right) {
  if (left.sign != right.sign) {
    return false;
  }
  Decimal l(left);
  Decimal r(right);
  if (l.frac != r.frac) {
    l.AlignAddSubData(r);
  }
  // #pragma unroll
  for (int i = 0; i < NUM_TOTAL_DIG; i++) {
    if (l.v[i] != r.v[i]) {
      return false;
    }
  }
  return true;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                            const Decimal& right) {
  return !(left == right);
}

// for int8_t
ARIES_HOST_DEVICE_NO_INLINE bool operator>(int8_t left, const Decimal& right) {
  return (int32_t) left > right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int8_t left, const Decimal& right) {
  return (int32_t) left >= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(int8_t left, const Decimal& right) {
  return (int32_t) left < right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int8_t left, const Decimal& right) {
  return (int32_t) left <= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(int8_t left, const Decimal& right) {
  return (int32_t) left == right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int8_t left, const Decimal& right) {
  return !(left == right);
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left, int8_t right) {
  return left > (int32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left, int8_t right) {
  return left >= (int32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left, int8_t right) {
  return left < (int32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left, int8_t right) {
  return left <= (int32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left, int8_t right) {
  return left == (int32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left, int8_t right) {
  return left != (int32_t) right;
}

// for uint8_t
ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint8_t left, const Decimal& right) {
  return (uint32_t) left > right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint8_t left,
                                            const Decimal& right) {
  return (uint32_t) left >= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint8_t left, const Decimal& right) {
  return (uint32_t) left < right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint8_t left,
                                            const Decimal& right) {
  return (uint32_t) left <= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint8_t left,
                                            const Decimal& right) {
  return (uint32_t) left == right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint8_t left,
                                            const Decimal& right) {
  return !(left == right);
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left, uint8_t right) {
  return left > (uint32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                            uint8_t right) {
  return left >= (uint32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left, uint8_t right) {
  return left < (uint32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                            uint8_t right) {
  return left <= (uint32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                            uint8_t right) {
  return left == (uint32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                            uint8_t right) {
  return left != (uint32_t) right;
}

// for int16_t
ARIES_HOST_DEVICE_NO_INLINE bool operator>(int16_t left, const Decimal& right) {
  return (int32_t) left > right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int16_t left,
                                            const Decimal& right) {
  return (int32_t) left >= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(int16_t left, const Decimal& right) {
  return (int32_t) left < right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int16_t left,
                                            const Decimal& right) {
  return (int32_t) left <= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(int16_t left,
                                            const Decimal& right) {
  return (int32_t) left == right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int16_t left,
                                            const Decimal& right) {
  return (int32_t) left != right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left, int16_t right) {
  return left > (int32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                            int16_t right) {
  return left >= (int32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left, int16_t right) {
  return left < (int32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                            int16_t right) {
  return left <= (int32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                            int16_t right) {
  return left == (int32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                            int16_t right) {
  return left != (int32_t) right;
}

// for uint16_t
ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint16_t left,
                                           const Decimal& right) {
  return (uint32_t) left > right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint16_t left,
                                            const Decimal& right) {
  return (uint32_t) left >= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint16_t left,
                                           const Decimal& right) {
  return (uint32_t) left < right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint16_t left,
                                            const Decimal& right) {
  return (uint32_t) left <= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint16_t left,
                                            const Decimal& right) {
  return (uint32_t) left == right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint16_t left,
                                            const Decimal& right) {
  return (uint32_t) left != right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                           uint16_t right) {
  return left > (uint32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                            uint16_t right) {
  return left >= (uint32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                           uint16_t right) {
  return left < (uint32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                            uint16_t right) {
  return left <= (uint32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                            uint16_t right) {
  return left == (uint32_t) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                            uint16_t right) {
  return left != (uint32_t) right;
}

// for int32_t
ARIES_HOST_DEVICE_NO_INLINE bool operator>(int32_t left, const Decimal& right) {
  Decimal d(left);
  return d > right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int32_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d >= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(int32_t left, const Decimal& right) {
  Decimal d(left);
  return d < right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int32_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d <= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(int32_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d == right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int32_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d != right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left, int32_t right) {
  Decimal d(right);
  return left > d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                            int32_t right) {
  Decimal d(right);
  return left >= d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left, int32_t right) {
  Decimal d(right);
  return left < d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                            int32_t right) {
  Decimal d(right);
  return left <= d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                            int32_t right) {
  Decimal d(right);
  return left == d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                            int32_t right) {
  Decimal d(right);
  return left != d;
}

// for uint32_t
ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint32_t left,
                                           const Decimal& right) {
  Decimal d(left);
  return d > right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint32_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d >= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint32_t left,
                                           const Decimal& right) {
  Decimal d(left);
  return d < right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint32_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d <= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint32_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d == right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint32_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d != right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                           uint32_t right) {
  Decimal d(right);
  return left > d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                            uint32_t right) {
  Decimal d(right);
  return left >= d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                           uint32_t right) {
  Decimal d(right);
  return left < d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                            uint32_t right) {
  Decimal d(right);
  return left <= d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                            uint32_t right) {
  Decimal d(right);
  return left == d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                            uint32_t right) {
  Decimal d(right);
  return left != d;
}

// for int64_t
ARIES_HOST_DEVICE_NO_INLINE bool operator>(int64_t left, const Decimal& right) {
  Decimal d(left);
  return d > right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(int64_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d >= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(int64_t left, const Decimal& right) {
  Decimal d(left);
  return d < right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(int64_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d <= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(int64_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d == right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(int64_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d != right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left, int64_t right) {
  Decimal d(right);
  return left > d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                            int64_t right) {
  Decimal d(right);
  return left >= d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left, int64_t right) {
  Decimal d(right);
  return left < d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                            int64_t right) {
  Decimal d(right);
  return left <= d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                            int64_t right) {
  Decimal d(right);
  return left == d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                            int64_t right) {
  Decimal d(right);
  return left != d;
}

// for uint64_t
ARIES_HOST_DEVICE_NO_INLINE bool operator>(uint64_t left,
                                           const Decimal& right) {
  Decimal d(left);
  return d > right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(uint64_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d >= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(uint64_t left,
                                           const Decimal& right) {
  Decimal d(left);
  return d < right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(uint64_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d <= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(uint64_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d == right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(uint64_t left,
                                            const Decimal& right) {
  Decimal d(left);
  return d != right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left,
                                           uint64_t right) {
  Decimal d(right);
  return left > d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left,
                                            uint64_t right) {
  Decimal d(right);
  return left >= d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left,
                                           uint64_t right) {
  Decimal d(right);
  return left < d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left,
                                            uint64_t right) {
  Decimal d(right);
  return left <= d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left,
                                            uint64_t right) {
  Decimal d(right);
  return left == d;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left,
                                            uint64_t right) {
  Decimal d(right);
  return left != d;
}

// for float
ARIES_HOST_DEVICE_NO_INLINE bool operator>(float left, const Decimal& right) {
  return (double) left > right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(float left, const Decimal& right) {
  return (double) left >= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(float left, const Decimal& right) {
  return (double) left < right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(float left, const Decimal& right) {
  return (double) left <= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(float left, const Decimal& right) {
  return (double) left == right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(float left, const Decimal& right) {
  return (double) left != right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left, float right) {
  return left > (double) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left, float right) {
  return left >= (double) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left, float right) {
  return left < (double) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left, float right) {
  return left <= (double) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left, float right) {
  return left == (double) right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left, float right) {
  return left != (double) right;
}

// for double
ARIES_HOST_DEVICE_NO_INLINE bool operator>(double left, const Decimal& right) {
  return left > right.GetDouble();
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(double left, const Decimal& right) {
  return left >= right.GetDouble();
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(double left, const Decimal& right) {
  return left < right.GetDouble();
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(double left, const Decimal& right) {
  return left <= right.GetDouble();
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(double left, const Decimal& right) {
  return left == right.GetDouble();
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(double left, const Decimal& right) {
  return left != right.GetDouble();
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>(const Decimal& left, double right) {
  return left.GetDouble() > right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator>=(const Decimal& left, double right) {
  return left.GetDouble() >= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<(const Decimal& left, double right) {
  return left.GetDouble() < right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator<=(const Decimal& left, double right) {
  return left.GetDouble() <= right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator==(const Decimal& left, double right) {
  return left.GetDouble() == right;
}

ARIES_HOST_DEVICE_NO_INLINE bool operator!=(const Decimal& left, double right) {
  return left.GetDouble() != right;
}

ARIES_HOST_DEVICE_NO_INLINE void Decimal::AddDecimalOnCpu(const Decimal& d) {
  Decimal added(d);
  added.AlignAddSubData(*this);
  uint64_t temp;
  uint32_t carry = 0;
  if (sign == added.sign) {
    // #pragma unroll
    for (int32_t i = 0; i < NUM_TOTAL_DIG; i++) {
      temp = (uint64_t) v[i] + added.v[i] + carry;
      v[i] = temp & 0x00000000ffffffff;
      carry = temp >> 32;
    }
  } else {
    int64_t r = 0;
    // #pragma unroll
    for (int32_t i = NUM_TOTAL_DIG - 1; i >= 0; i--) {
      r = (int64_t) v[i] - added.v[i];
      if (r != 0) {
        break;
      }
    }
    if (r >= 0) {
      // #pragma unroll
      for (int32_t i = 0; i < NUM_TOTAL_DIG; i++) {
        temp = (uint64_t) v[i] + PER_DEC_MAX_SCALE - added.v[i] - carry;
        carry =
            (temp < PER_DEC_MAX_SCALE);  // 比 PER_DEC_MAX_SCALE 小表示借位了
        v[i] = temp & 0x00000000ffffffff;  // 对 temp 取模
      }
    } else {
      // #pragma unroll
      for (int32_t i = 0; i < NUM_TOTAL_DIG; i++) {
        temp = (uint64_t) added.v[i] + PER_DEC_MAX_SCALE - v[i] - carry;
        carry =
            (temp < PER_DEC_MAX_SCALE);  // 比 PER_DEC_MAX_SCALE 小表示借位了
        v[i] = temp & 0x00000000ffffffff;
      }
    }
    sign = (r > 0 && !d.sign) || (r < 0 && d.sign);
  }
}

ARIES_HOST_DEVICE_NO_INLINE void Decimal::MulDecimalOnCpu(const Decimal& d) {
  sign = d.sign ^ sign;
  frac = frac + d.frac;

  uint32_t inner_res[NUM_TOTAL_DIG * 2] = {0};

  uint64_t temp;
  uint32_t carry;

  // #pragma unroll
  for (int i = 0; i < NUM_TOTAL_DIG; i++) {
    carry = 0;
    // #pragma unroll
    for (int j = 0; j < NUM_TOTAL_DIG; j++) {
      temp = (uint64_t) v[i] * d.v[j] + inner_res[i + j] + carry;
      carry = temp / PER_DEC_MAX_SCALE;
      inner_res[i + j] = temp % PER_DEC_MAX_SCALE;
    }
    inner_res[i + NUM_TOTAL_DIG] = carry;
  }

  // #pragma unroll
  for (int i = INDEX_LAST_DIG; i >= 0; i--) {
    v[i] = inner_res[i];
  }
}

// 计算加法目标精度，生成的动态代码需要此数
ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcAddTargetPrecision(
    const Decimal& d) {
  prec = aries_max(prec - frac, d.prec - d.frac) + 1;
  frac = aries_max(frac, d.frac);
  prec += frac;
}

// 加法函数：decimalx += decimaly 函数
ARIES_DEVICE Decimal& Decimal::operator+=(const Decimal& d) {
  Decimal added(d);
  added.AlignAddSubData(*this);

  if (sign == added.sign) {
    asm volatile("add.cc.u32 %0, %1, %2;"
                 : "=r"(v[0])
                 : "r"(added.v[0]), "r"(v[0]));
    // #pragma unroll
    for (int32_t i = 1; i < NUM_TOTAL_DIG; i++)
      asm volatile("addc.cc.u32 %0, %1, %2;"
                   : "=r"(v[i])
                   : "r"(added.v[i]), "r"(v[i]));
  } else {
    int64_t r = 0;
    // #pragma unroll
    for (int i = NUM_TOTAL_DIG - 1; i >= 0; i--) {
      r = (int64_t) v[i] - added.v[i];
      if (r != 0) {
        break;
      }
    }
    if (r >= 0) {
      asm volatile("sub.cc.u32 %0, %1, %2;"
                   : "=r"(v[0])
                   : "r"(v[0]), "r"(added.v[0]));
      // #pragma unroll
      for (int32_t i = 1; i < NUM_TOTAL_DIG; i++)
        asm volatile("subc.cc.u32 %0, %1, %2;"
                     : "=r"(v[i])
                     : "r"(v[i]), "r"(added.v[i]));
    } else {
      asm volatile("sub.cc.u32 %0, %1, %2;"
                   : "=r"(v[0])
                   : "r"(added.v[0]), "r"(v[0]));
      // #pragma unroll
      for (int32_t i = 1; i < NUM_TOTAL_DIG; i++)
        asm volatile("subc.cc.u32 %0, %1, %2;"
                     : "=r"(v[i])
                     : "r"(added.v[i]), "r"(v[i]));
    }
    sign = (r > 0 && !d.sign) || (r < 0 && d.sign);
  }
  return *this;
}

// signed
ARIES_DEVICE Decimal& Decimal::operator+=(int8_t i) {
  Decimal d(i);
  return *this += d;
}

ARIES_DEVICE Decimal& Decimal::operator+=(int16_t i) {
  Decimal d(i);
  return *this += d;
}

ARIES_DEVICE Decimal& Decimal::operator+=(int32_t i) {
  Decimal d(i);
  return *this += d;
}

ARIES_DEVICE Decimal& Decimal::operator+=(int64_t i) {
  Decimal d(i);
  return *this += d;
}

// unsigned
ARIES_DEVICE Decimal& Decimal::operator+=(uint8_t i) {
  Decimal d(i);
  return *this += d;
}

ARIES_DEVICE Decimal& Decimal::operator+=(uint16_t i) {
  Decimal d(i);
  return *this += d;
}

ARIES_DEVICE Decimal& Decimal::operator+=(uint32_t i) {
  Decimal d(i);
  return *this += d;
}

ARIES_DEVICE Decimal& Decimal::operator+=(uint64_t i) {
  Decimal d(i);
  return *this += d;
}

// double / float
ARIES_DEVICE double Decimal::operator+=(const float& f) {
  return *this += (double) f;
}

ARIES_DEVICE double Decimal::operator+=(const double& l) {
  return GetDouble() + l;
}

// self operator
ARIES_DEVICE Decimal& Decimal::operator++() {
  Decimal d((int8_t) 1);
  *this += d;
  return *this;
}

ARIES_DEVICE Decimal Decimal::operator++(int32_t) {
  Decimal d((int8_t) 1);
  *this += d;
  return *this;
}

// 加法函数，decimalx + decimaly
ARIES_DEVICE Decimal operator+(const Decimal& left, const Decimal& right) {
  // 将 const left 赋值到temp进行操作
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(const Decimal& left, int8_t right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(const Decimal& left, int16_t right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(const Decimal& left, int32_t right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(const Decimal& left, int64_t right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(int8_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(int16_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(int32_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(int64_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp += right;
}

// unsigned
ARIES_DEVICE Decimal operator+(const Decimal& left, uint8_t right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(const Decimal& left, uint16_t right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(const Decimal& left, uint32_t right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(const Decimal& left, uint64_t right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(uint8_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(uint16_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(uint32_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp += right;
}

ARIES_DEVICE Decimal operator+(uint64_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp += right;
}

// double / float
ARIES_DEVICE double operator+(const Decimal& left, float right) {
  return left.GetDouble() + right;
}

ARIES_DEVICE double operator+(const Decimal& left, double right) {
  return left.GetDouble() + right;
}

ARIES_DEVICE double operator+(float left, const Decimal& right) {
  return left + right.GetDouble();
}

ARIES_DEVICE double operator+(double left, const Decimal& right) {
  return left + right.GetDouble();
}

// 计算目标结果精度，在SQL转化为动态代码时用到
ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcSubTargetPrecision(
    const Decimal& d) {
  prec = aries_max(prec - frac, d.prec - d.frac) + 1;
  frac = aries_max(frac, d.frac);
  prec += frac;
}

// 减法函数 decimalx -= decimaly
ARIES_DEVICE Decimal& Decimal::operator-=(const Decimal& d) {
  Decimal added(d);
  added.AlignAddSubData(*this);

  if (added.sign != sign) {
    asm volatile("add.cc.u32 %0, %1, %2;"
                 : "=r"(v[0])
                 : "r"(added.v[0]), "r"(v[0]));

    // #pragma unroll
    for (int32_t i = 1; i < NUM_TOTAL_DIG; i++)
      asm volatile("addc.cc.u32 %0, %1, %2;"
                   : "=r"(v[i])
                   : "r"(added.v[i]), "r"(v[i]));

  } else {
    int64_t r = 0;
    // #pragma unroll
    for (int i = NUM_TOTAL_DIG - 1; i >= 0; i--) {
      r = (int64_t) v[i] - added.v[i];
      if (r != 0) {
        break;
      }
    }
    if (r >= 0) {
      asm volatile("sub.cc.u32 %0, %1, %2;"
                   : "=r"(v[0])
                   : "r"(v[0]), "r"(added.v[0]));
      // #pragma unroll
      for (int32_t i = 1; i < NUM_TOTAL_DIG; i++)
        asm volatile("subc.cc.u32 %0, %1, %2;"
                     : "=r"(v[i])
                     : "r"(v[i]), "r"(added.v[i]));
    } else {
      asm volatile("sub.cc.u32 %0, %1, %2;"
                   : "=r"(v[0])
                   : "r"(added.v[0]), "r"(v[0]));
      // #pragma unroll
      for (int32_t i = 1; i < NUM_TOTAL_DIG; i++)
        asm volatile("subc.cc.u32 %0, %1, %2;"
                     : "=r"(v[i])
                     : "r"(added.v[i]), "r"(v[i]));
    }
    sign = (r > 0 && added.sign) || (r < 0 && !added.sign);
  }

  return *this;
}

// signed
ARIES_DEVICE Decimal& Decimal::operator-=(int8_t i) {
  Decimal d(i);
  return *this -= d;
}

ARIES_DEVICE Decimal& Decimal::operator-=(int16_t i) {
  Decimal d(i);
  return *this -= d;
}

ARIES_DEVICE Decimal& Decimal::operator-=(int32_t i) {
  Decimal d(i);
  return *this -= d;
}

ARIES_DEVICE Decimal& Decimal::operator-=(int64_t i) {
  Decimal d(i);
  return *this -= d;
}

// unsigned
ARIES_DEVICE Decimal& Decimal::operator-=(uint8_t i) {
  Decimal d(i);
  return *this -= d;
}

ARIES_DEVICE Decimal& Decimal::operator-=(uint16_t i) {
  Decimal d(i);
  return *this -= d;
}

ARIES_DEVICE Decimal& Decimal::operator-=(uint32_t i) {
  Decimal d(i);
  return *this -= d;
}

ARIES_DEVICE Decimal& Decimal::operator-=(uint64_t i) {
  Decimal d(i);
  return *this -= d;
}

// double / float
ARIES_DEVICE double Decimal::operator-=(const float& f) {
  return GetDouble() - f;
}

ARIES_DEVICE double Decimal::operator-=(const double& l) {
  return GetDouble() - l;
}

// self operator
ARIES_DEVICE Decimal& Decimal::operator--() {
  Decimal d((int8_t) 1);
  return *this -= d;
}

ARIES_DEVICE Decimal Decimal::operator--(int32_t) {
  Decimal tmp(*this);
  Decimal d((int8_t) 1);
  return tmp -= d;
}

// 减法函数，decimalx - decimaly
ARIES_DEVICE Decimal operator-(const Decimal& left, const Decimal& right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(const Decimal& left, int8_t right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(const Decimal& left, int16_t right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(const Decimal& left, int32_t right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(const Decimal& left, int64_t right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(int8_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(int16_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp -= right;
}

// 减法函数，decimalx -= decimaly
ARIES_DEVICE Decimal operator-(int32_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(int64_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp -= right;
}

// unsigned
ARIES_DEVICE Decimal operator-(const Decimal& left, uint8_t right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(const Decimal& left, uint16_t right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(const Decimal& left, uint32_t right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(const Decimal& left, uint64_t right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(uint8_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(uint16_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(uint32_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp -= right;
}

ARIES_DEVICE Decimal operator-(uint64_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp -= right;
}

// double / float
ARIES_DEVICE double operator-(const Decimal& left, const float right) {
  return left.GetDouble() - right;
}

ARIES_DEVICE double operator-(const Decimal& left, const double right) {
  return left.GetDouble() - right;
}

ARIES_DEVICE double operator-(const float left, const Decimal& right) {
  return left - right.GetDouble();
}

ARIES_DEVICE double operator-(const double left, const Decimal& right) {
  return left - right.GetDouble();
}

// 计算乘法目标精度
ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcMulTargetPrecision(
    const Decimal& d) {
  frac = frac + d.frac;
  prec = prec + d.prec;
}

// 乘法函数 decimalx *= decimaly
ARIES_DEVICE Decimal& Decimal::operator*=(const Decimal& d) {
  sign = d.sign ^ sign;
  frac = frac + d.frac;

  uint32_t inner_res[NUM_TOTAL_DIG * 2] = {0};

  uint64_t temp;
  uint32_t carry;

  // #pragma unroll
  for (int i = 0; i < NUM_TOTAL_DIG; i++) {
    carry = 0;
    // #pragma unroll
    for (int j = 0; j < NUM_TOTAL_DIG; j++) {
      // temp 表示范围最大值 2^64-1 右侧表达式 表示范围最大值 (2^32-1) *
      // (2^32-1) + (2^32-1) + (2^32-1) = 2^64-1
      temp = (uint64_t) v[i] * d.v[j] + inner_res[i + j] + carry;
      carry = temp / PER_DEC_MAX_SCALE;
      inner_res[i + j] = temp % PER_DEC_MAX_SCALE;
    }
    inner_res[i + NUM_TOTAL_DIG] = carry;
  }

  // #pragma unroll
  for (int i = INDEX_LAST_DIG; i >= 0; i--) {
    v[i] = inner_res[i];
  }

  return *this;
}

// signed
ARIES_DEVICE Decimal& Decimal::operator*=(int8_t i) {
  Decimal tmp(i);
  return *this *= tmp;
}

ARIES_DEVICE Decimal& Decimal::operator*=(int16_t i) {
  Decimal tmp(i);
  return *this *= tmp;
}

ARIES_DEVICE Decimal& Decimal::operator*=(int32_t i) {
  Decimal tmp(i);
  return *this *= tmp;
}

ARIES_DEVICE Decimal& Decimal::operator*=(int64_t i) {
  Decimal tmp(i);
  return *this *= tmp;
}

// unsigned
ARIES_DEVICE Decimal& Decimal::operator*=(uint8_t i) {
  Decimal tmp(i);
  return *this *= tmp;
}

ARIES_DEVICE Decimal& Decimal::operator*=(uint16_t i) {
  Decimal tmp(i);
  return *this *= tmp;
}

ARIES_DEVICE Decimal& Decimal::operator*=(uint32_t i) {
  Decimal tmp(i);
  return *this *= tmp;
}

ARIES_DEVICE Decimal& Decimal::operator*=(uint64_t i) {
  Decimal tmp(i);
  return *this *= tmp;
}

// double / float
ARIES_DEVICE double Decimal::operator*=(const float& f) {
  return GetDouble() * f;
}

ARIES_DEVICE double Decimal::operator*=(const double& d) {
  return GetDouble() * d;
}

// 乘法运算 decimalx * decimaly
ARIES_DEVICE Decimal operator*(const Decimal& left, const Decimal& right) {
  Decimal tmp(left);
  return tmp *= right;
}

// signed
ARIES_DEVICE Decimal operator*(const Decimal& left, int8_t right) {
  Decimal tmp(right);
  return tmp *= left;
}

ARIES_DEVICE Decimal operator*(const Decimal& left, int16_t right) {
  Decimal tmp(right);
  return tmp *= left;
}

ARIES_DEVICE Decimal operator*(const Decimal& left, int32_t right) {
  Decimal tmp(right);
  return tmp *= left;
}

ARIES_DEVICE Decimal operator*(const Decimal& left, int64_t right) {
  Decimal tmp(right);
  return tmp *= left;
}

ARIES_DEVICE Decimal operator*(int8_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp *= right;
}

ARIES_DEVICE Decimal operator*(int16_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp *= right;
}

ARIES_DEVICE Decimal operator*(int32_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp *= right;
}

ARIES_DEVICE Decimal operator*(int64_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp *= right;
}

// unsigned
ARIES_DEVICE Decimal operator*(const Decimal& left, uint8_t right) {
  Decimal tmp(right);
  return tmp *= left;
}

ARIES_DEVICE Decimal operator*(const Decimal& left, uint16_t right) {
  Decimal tmp(right);
  return tmp *= left;
}

ARIES_DEVICE Decimal operator*(const Decimal& left, uint32_t right) {
  Decimal tmp(right);
  return tmp *= left;
}

ARIES_DEVICE Decimal operator*(const Decimal& left, uint64_t right) {
  Decimal tmp(right);
  return tmp *= left;
}

ARIES_DEVICE Decimal operator*(uint8_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp *= right;
}

ARIES_DEVICE Decimal operator*(uint16_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp *= right;
}

ARIES_DEVICE Decimal operator*(uint32_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp *= right;
}

ARIES_DEVICE Decimal operator*(uint64_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp *= right;
}

// double / float
ARIES_DEVICE double operator*(const Decimal& left, const float right) {
  return left.GetDouble() * right;
}

ARIES_DEVICE double operator*(const Decimal& left, const double right) {
  return left.GetDouble() * right;
}

ARIES_DEVICE double operator*(const float left, const Decimal& right) {
  return left * right.GetDouble();
}

ARIES_DEVICE double operator*(const double left, const Decimal& right) {
  return left * right.GetDouble();
}

// 除法目标精度
ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcDivTargetPrecision(
    const Decimal& d) {
  prec = (prec - frac) + d.frac;
  frac = frac + DIV_FIX_EX_FRAC;
  prec += frac;
}

// 右移 n个 10 进制位 TODO zzh
ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::operator>>(int n) {
  // 转化为循环除法
  uint64_t temp = 0;
  uint32_t remainder = 0;
  while (n > DIG_PER_INT32) {
    for (int i = INDEX_LAST_DIG; i >= 0; i--) {
      temp = remainder * PER_DEC_MAX_SCALE + v[i];
      v[i] = temp / MAX_BASE10_PER_INT;
      remainder = temp % MAX_BASE10_PER_INT;
    }
    n -= DIG_PER_INT32;
  }

  uint32_t pow10n = GetPowers10(n);
  remainder = 0;
  for (int i = INDEX_LAST_DIG; i >= 0; i--) {
    temp = remainder * PER_DEC_MAX_SCALE + v[i];
    v[i] = temp / pow10n;
    remainder = temp % pow10n;
  }

  return *this;
}

// 左移 n个 intBit
ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::operator<<(int n) {
  uint32_t res[NUM_TOTAL_DIG] = {0};
  uint32_t carry = 0;
  uint64_t temp;
  // #pragma unroll
  for (int i = 0; i < NUM_TOTAL_DIG; i++) {
    carry = 0;
    // #pragma unroll
    for (int j = 0; j < __ARRAY_SCALE[n - 1]; j++) {
      if (i + j > NUM_TOTAL_DIG) {
        break;
      }
      temp = (uint64_t) v[i] * (uint64_t) __POW10_ARRAY[n - 1][j] + res[i + j] +
             carry;
      carry = temp >> 32;
      res[i + j] = temp & 0x00000000ffffffff;
    }
    if (i + __ARRAY_SCALE[n - 1] < NUM_TOTAL_DIG) {
      res[i + __ARRAY_SCALE[n - 1]] = carry;
    }
  }
  // #pragma unroll
  for (int i = 0; i < NUM_TOTAL_DIG; i++) {
    v[i] = res[i];
  }
  return *this;
}

// 检查并设置位真实十进制下的精度 此处用的是 double
ARIES_HOST_DEVICE_NO_INLINE void Decimal::CheckAndSetRealPrecision() {
  int i = INDEX_LAST_DIG;
  for (; i >= 0; i--) {
    if (v[i] != 0) {
      break;
    }
  }
  if (i == -1) {
    prec = 1;
    frac = 0;
  } else {
    prec = i * DIG_PER_INT32;
    double maxUint32 = 4.294967295;
    double tmpMul = 1;
    for (int j = 0; j < i; j++) {
      tmpMul *= maxUint32;
    }
    tmpMul *= v[i];
    int tt = (int) tmpMul;
    while (tt > 0) {
      tt /= 10;
      prec++;
    }
  }
}

// 无用函数
ARIES_HOST_DEVICE_NO_INLINE Decimal& Decimal::GenIntDecimal(int shift) {
  return *this;
}

// 取中值函数，计算 (decimalx + decimaly) / 2
ARIES_DEVICE Decimal Decimal::HalfIntDecimal(const Decimal d1,
                                             const Decimal d2) {
  Decimal tmp(d1);
  tmp += d2;
  uint32_t rds = 0;
  uint64_t t[NUM_TOTAL_DIG];
  // #pragma unroll
  for (int i = 0; i < NUM_TOTAL_DIG; i++) {
    t[i] = tmp.v[i];
  }
  // 此时t[i]中存放qmax+qmin的值

  // #pragma unroll
  for (int i = INDEX_LAST_DIG; i >= 0; i--) {
    if (rds) {
      t[i] += rds * PER_DEC_MAX_SCALE;
    }
    if (t[i]) {
      rds = t[i] % 2;
      t[i] /= 2;
    }
  }
  // 上述过程将qmax+qmin的值除以2，并将值存放于t[i]之中
  // #pragma unroll
  for (int i = 0; i < NUM_TOTAL_DIG; i++) {
    tmp.v[i] = t[i];
  }
  return tmp;
}

// DivInt函数，计算两个数相除 decimalx / decimal y 时使用
// residuel 是余数
ARIES_DEVICE Decimal Decimal::DivInt(const Decimal ds, const Decimal dt,
                                     Decimal& residuel, uint32_t dtHitBit) {
  // 被除数为0
  if (ds.isZero()) {
    aries_memset(residuel.v, 0x00, sizeof(residuel.v));
    return ds;
  }
  // 通过二进制的方法得出数据上下限
  uint32_t dsz = INDEX_LAST_DIG;
  while (ds.v[dsz] == 0)
    dsz--;
  // 被除数 的最高位的 bit 所在的位置
  uint32_t dsHitBit = 0;
  asm volatile("bfind.u32 %0, %1;" : "=r"(dsHitBit) : "r"(ds.v[dsz]));
  dsHitBit++;
  // 被除数的最高位 是 dsHitBit
  dsHitBit += dsz * PER_INT_MAX_BIT;

  // 被除数 最高位的 bit 所在位置是 dsHitBit 所以被除数表示的范围是 [
  // 2^(dsHitBit-1) , 2^dsHitBit ) 除数的 最高位的 bit 所在位置是 dtHitBit
  // 所以除数表示的范围是 [ 2^(dtHitBit-1) , 2^dtHitBit ) 上限 2^dsHitBit /
  // 2^(dtHitBit-1) -> 2^(dsHitBit-dtHitBit+1) 下限 2^(dsHitBit-1) / 2^dtHitBit
  // -> 2^(dsHitBit-dtHitBit-1) 所以 上限是 pmax 下限是 pmin
  uint32_t pmax = dsHitBit - dtHitBit + 1;
  uint32_t pmin = dsHitBit - dtHitBit - 1;

  // 根据上下限构造数字
  Decimal qmax(0), qmin(0), qmid(0), restmp(0);
  // 为 qmax 赋值
  int pmax_index = pmax / PER_INT_MAX_BIT;
  qmax.v[pmax_index] = 1;
  qmax.v[pmax_index] <<= (pmax % PER_INT_MAX_BIT);
  // 为 qmin 赋值
  int pmin_index = pmin / PER_INT_MAX_BIT;
  qmin.v[pmin_index] = 1;
  qmin.v[pmin_index] <<= (pmin % PER_INT_MAX_BIT);

  // 采用二分法求值
  while (qmin < qmax) {
    // 取中值
    qmid = HalfIntDecimal(qmax, qmin);
    // 比较 qmid 和 qmin 的大小。若相等说明结果找到 就是qmid
    if (qmid == qmin) {
      break;
    }
    // 计算 中值 * 除数 与 被除数进行比较
    restmp = qmid * dt;
    // 如果 rsdtmp == ds说明被整除，直接返回
    if (restmp == ds) {
      qmin.CopyValue(qmid);
      break;
    } else if (restmp < ds) {
      // 如果为小于被除数，说明 商 在qmid ~ qmax 区间
      qmin = qmid;
    } else {
      // 如果为大于被除数，说明 商 在qmin ~ qmid 区间
      qmax = qmid;
    }
  }
  residuel = ds - qmin * dt;
  return qmin;
}

// 将 this v值转给 decimal d
ARIES_HOST_DEVICE_NO_INLINE void Decimal::CopyValue(Decimal& d) {
  // #pragma unroll
  for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
    v[i] = d.v[i];
  }
}

// DivOrMod函数
ARIES_DEVICE Decimal& Decimal::DivOrMod(const Decimal& d, bool isMod) {
  // 被除数 和 除数
  Decimal divitend(*this);
  Decimal divisor(d);

  // 对符号位进行判断
  sign = d.sign ^ sign;

  // 判断是否为mod
  if (isMod) {
  } else {
    // 除法操作 计算精度 遵循old.cu的精度 被除数的精度 + DIV_FIX_INNER_FRAC
    frac += DIV_FIX_EX_FRAC;
  }

  // 被除数为0，返回被除数，return *this返回的是对象本身
  if (isZero()) {
    sign = 0;
    prec = frac + 1;
    return *this;
  }

  // 除数为零时，error 标志位
  if (d.isZero()) {
    error = ERR_DIV_BY_ZERO;
    return *this;
  }

  // TODO 这里不需要 六位扩充
  // 因为保留了 被除数的精度 + DIV_FIX_INNER_FRAC 个 10 进制位， 所以右移动
  // 除数的精度 + DIV_FIX_INNER_FRAC 个 10 进制位
  uint32_t shift = divisor.frac + DIV_FIX_EX_FRAC;

  divitend.frac = 0;
  divisor.frac = 0;
  divitend.sign = 0;
  divisor.sign = 0;

  // 用res来存储计算结果
  Decimal res(0);

  uint32_t dtz = INDEX_LAST_DIG;
  uint32_t dsz = INDEX_LAST_DIG;
  while (divitend.v[dtz] == 0)
    dtz--;
  while (divisor.v[dsz] == 0)
    dsz--;

  // 被除数 的最高位的 bit 所在的位置 第一个位置是 0
  uint32_t hitDtBit = 0;
  asm volatile("bfind.u32 %0, %1;" : "=r"(hitDtBit) : "r"(divitend.v[dtz]));
  hitDtBit++;
  hitDtBit += dtz * PER_INT_MAX_BIT;

  // 左移 shift 位 相当于 被除数 乘以 shift 的 10 次方
  // 需要向左偏移的 shiftVal 的最高位
  // 此处 十进制的 10 100 1000 分别在 二进制下的位数是 4 3 3
  uint32_t hitShiftBit = (shift / 3) * 10;
  if (shift % 3 == 1)
    hitShiftBit += 4;
  else if (shift % 3 == 2)
    hitShiftBit += 7;

  // 除数 的最高位的 bit 所在的位置
  uint32_t hitDsBit = 0;
  asm volatile("bfind.u32 %0, %1;" : "=r"(hitDsBit) : "r"(divisor.v[dtz]));
  hitDsBit++;
  hitDsBit += dsz * PER_INT_MAX_BIT;

  // 对于unsign long类型数据的处理,这里的被除数都应该是unsign long类型数据
  // 被除数占用了hitDtBit位 最大值为 2^hitDtBit-1 左移的十进制数占用了
  // hitShiftBit 最大值位数为 2^hitShiftBit-1 可以表示为 2^(hitShiftBit-1)*1 +
  // 2^(hitShiftBit-2)*1 or 0 + …… 所以被除数左移后最高位 在 hitDtBit +
  // hitShiftBit - 1  这个位置上
  if (hitDtBit + hitShiftBit - 1 <= PER_INT_MAX_BIT * 2 &&
      hitDsBit <= PER_INT_MAX_BIT * 2) {
    res = divitend.DivByInt64(divisor, shift, isMod);
  }
  // 对于unsign int类型数据的处理,这里的除数都应该是unsign int类型数据
  else if (dsz == 0) {
    res = divitend.DivByInt(divisor, shift, isMod);
  } else {
    // 二分计算;
    // 待左移的量
    int tmpEx = shift;
    // 左移量 因为不能一下子完成左移 可能需要分多次
    int nDigits = 0;
    // tmpRes保存中间结果
    Decimal tmpRes(0);
    for (; tmpEx > 0;) {
      // 这样能算出一次性最多左移的大小-1此处遵循old
      //  最大可左移的 10 的倍数
      //  这里缩小范围  一个 Uint32 最大可以左移 9
      nDigits = (INDEX_LAST_DIG - dtz + 1) * DIG_PER_INT32;
      uint32_t tmp = divitend.v[dtz];
      // 那么最大可 左移（十进制）的数量是 没用到的uint32 * 9 + 9 -用到的第一个
      // uint32 已用的十进制空间
      while (tmp > 0) {
        tmp /= 10;
        nDigits--;
      }
      // 可左移的量 比 待左移的量大
      if (nDigits > tmpEx) {
        nDigits = tmpEx;
      }
      // 此次左移 nDigits
      tmpEx -= nDigits;
      // 左移
      divitend << nDigits;
      // 除法
      tmpRes = DivInt(divitend, divisor, divitend, hitDsBit);
      if (!res.isZero()) {
        // res左移nDigits位
        res << nDigits;
      }
      res += tmpRes;
    }
    // 四舍五入
    if (isMod) {
      res = divitend;
    } else {
      // 进行四舍五入
      Decimal doubleDivitend = divitend + divitend;
      if (doubleDivitend >= divisor) {
        asm volatile("add.cc.u32 %0, %1, %2;"
                     : "=r"(res.v[0])
                     : "r"(res.v[0]), "r"(1));
        // #pragma unroll
        for (int32_t i = 1; i < NUM_TOTAL_DIG; i++)
          asm volatile("addc.cc.u32 %0, %1, %2;"
                       : "=r"(res.v[i])
                       : "r"(res.v[i]), "r"(0));
      }
    }
  }
  CopyValue(res);
  return *this;
}

// int32_t除法
ARIES_DEVICE Decimal& Decimal::DivByInt(const Decimal& d, int shift,
                                        bool isMod) {
  // 存放除数
  uint32_t dvs = d.v[0];
  // 存放余数
  uint32_t remainder = 0;
  // 左移 shift 位数
  *this << shift;
  for (int i = NUM_TOTAL_DIG - 1; i >= 0; i--) {
    if (v[i] || remainder) {
      uint64_t tmp = (uint64_t) v[i] + (uint64_t) remainder * PER_DEC_MAX_SCALE;
      v[i] = tmp / dvs;
      remainder = tmp % dvs;
    }
  }
  // 四舍五入
  if (isMod) {
    // *this = remainder;
  } else {
    if (remainder * 2 >= dvs) {
      asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(v[0]) : "r"(v[0]), "r"(1));
      // #pragma unroll
      for (int32_t i = 1; i < NUM_TOTAL_DIG; i++)
        asm volatile("addc.cc.u32 %0, %1, %2;"
                     : "=r"(v[i])
                     : "r"(v[i]), "r"(0));
    }
  }
  return *this;
}

// int64相除
ARIES_DEVICE Decimal& Decimal::DivByInt64(const Decimal& divisor, int shift,
                                          bool isMod) {
  // 被除数 在 int64 范围内
  uint64_t dvs = ToUint64();
  while (shift > DIG_PER_INT32) {
    dvs *= GetPowers10(DIG_PER_INT32);
    shift -= DIG_PER_INT32;
  }
  dvs *= GetPowers10(shift);
  // 被除数
  uint64_t dvt = divisor.ToUint64();
  uint64_t res =
      isMod ? (dvs % dvt) : (dvs / dvt + (((dvs % dvt) << 1) >= dvt ? 1 : 0));

  v[1] = res / PER_DEC_MAX_SCALE;
  v[0] = res % PER_DEC_MAX_SCALE;
  return *this;
}

// 除法函数，decimalx /= decimaly TODO zzh
ARIES_DEVICE Decimal& Decimal::operator/=(const Decimal& d) {
  return DivOrMod(d);
}

// signed
ARIES_DEVICE Decimal& Decimal::operator/=(int8_t i) {
  Decimal d(i);
  return *this /= d;
}

ARIES_DEVICE Decimal& Decimal::operator/=(int16_t i) {
  Decimal d(i);
  return *this /= d;
}

ARIES_DEVICE Decimal& Decimal::operator/=(int32_t i) {
  Decimal d(i);
  return *this /= d;
}

ARIES_DEVICE Decimal& Decimal::operator/=(int64_t i) {
  Decimal d(i);
  return *this /= d;
}

// unsigned
ARIES_DEVICE Decimal& Decimal::operator/=(uint8_t i) {
  Decimal d(i);
  return *this /= d;
}

ARIES_DEVICE Decimal& Decimal::operator/=(uint16_t i) {
  Decimal d(i);
  return *this /= d;
}

ARIES_DEVICE Decimal& Decimal::operator/=(uint32_t i) {
  Decimal d(i);
  return *this /= d;
}

ARIES_DEVICE Decimal& Decimal::operator/=(uint64_t i) {
  Decimal d(i);
  return *this /= d;
}

// double / float
ARIES_DEVICE double Decimal::operator/=(const float& f) {
  return GetDouble() / f;
}

ARIES_DEVICE double Decimal::operator/=(const double& d) {
  return GetDouble() / d;
}

// 除法函数，decimalx / decimaly
ARIES_DEVICE Decimal operator/(const Decimal& left, const Decimal& right) {
  Decimal tmp(left);
  return tmp /= right;
}

// signed
ARIES_DEVICE Decimal operator/(const Decimal& left, int8_t right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(const Decimal& left, int16_t right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(const Decimal& left, int32_t right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(const Decimal& left, int64_t right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(int8_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(int16_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(int32_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(int64_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp /= right;
}

// unsigned
ARIES_DEVICE Decimal operator/(const Decimal& left, uint8_t right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(const Decimal& left, uint16_t right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(const Decimal& left, uint32_t right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(const Decimal& left, uint64_t right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(uint8_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(uint16_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(uint32_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp /= right;
}

ARIES_DEVICE Decimal operator/(uint64_t left, const Decimal& right) {
  Decimal tmp(left);
  return tmp /= right;
}

// double / float
ARIES_DEVICE double operator/(const Decimal& left, const float right) {
  return left.GetDouble() / right;
}

ARIES_DEVICE double operator/(const Decimal& left, const double right) {
  return left.GetDouble() / right;
}

ARIES_DEVICE double operator/(const float left, const Decimal& right) {
  return left / right.GetDouble();
}

ARIES_DEVICE double operator/(const double left, const Decimal& right) {
  return left / right.GetDouble();
}

// 计算取余的目标精度
ARIES_HOST_DEVICE_NO_INLINE void Decimal::CalcModTargetPrecision(
    const Decimal& d) {}

// 取余操作
ARIES_DEVICE Decimal& Decimal::operator%=(const Decimal& d) { return *this; }
// signed
ARIES_DEVICE Decimal& Decimal::operator%=(int8_t i) { return *this; }

ARIES_DEVICE Decimal& Decimal::operator%=(int16_t i) { return *this; }

ARIES_DEVICE Decimal& Decimal::operator%=(int32_t i) { return *this; }

ARIES_DEVICE Decimal& Decimal::operator%=(int64_t i) { return *this; }

// unsigned
ARIES_DEVICE Decimal& Decimal::operator%=(uint8_t i) { return *this; }

ARIES_DEVICE Decimal& Decimal::operator%=(uint16_t i) { return *this; }

ARIES_DEVICE Decimal& Decimal::operator%=(uint32_t i) { return *this; }

ARIES_DEVICE Decimal& Decimal::operator%=(uint64_t i) { return *this; }

// double % float
ARIES_DEVICE double Decimal::operator%=(const float& f) { return 0.0; }

ARIES_DEVICE double Decimal::operator%=(const double& d) { return 0.0; }

// two operators
ARIES_DEVICE Decimal operator%(const Decimal& left, const Decimal& right) {
  return left;
}

// signed
ARIES_DEVICE Decimal operator%(const Decimal& left, int8_t right) {
  return left;
}

ARIES_DEVICE Decimal operator%(const Decimal& left, int16_t right) {
  return left;
}

ARIES_DEVICE Decimal operator%(const Decimal& left, int32_t right) {
  return left;
}

ARIES_DEVICE Decimal operator%(const Decimal& left, int64_t right) {
  return left;
}

ARIES_DEVICE Decimal operator%(int8_t left, const Decimal& right) {
  return right;
}

ARIES_DEVICE Decimal operator%(int16_t left, const Decimal& right) {
  return right;
}

ARIES_DEVICE Decimal operator%(int32_t left, const Decimal& right) {
  return right;
}

ARIES_DEVICE Decimal operator%(int64_t left, const Decimal& right) {
  return right;
}

// unsigned
ARIES_DEVICE Decimal operator%(const Decimal& left, uint8_t right) {
  return left;
}

ARIES_DEVICE Decimal operator%(const Decimal& left, uint16_t right) {
  return left;
}

ARIES_DEVICE Decimal operator%(const Decimal& left, uint32_t right) {
  return left;
}

ARIES_DEVICE Decimal operator%(const Decimal& left, uint64_t right) {
  return left;
}

ARIES_DEVICE Decimal operator%(uint8_t left, const Decimal& right) {
  return right;
}

ARIES_DEVICE Decimal operator%(uint16_t left, const Decimal& right) {
  return right;
}

ARIES_DEVICE Decimal operator%(uint32_t left, const Decimal& right) {
  return right;
}

ARIES_DEVICE Decimal operator%(uint64_t left, const Decimal& right) {
  return right;
}

// double % float
ARIES_DEVICE double operator%(const Decimal& left, const float right) {
  return 0.0;
}

ARIES_DEVICE double operator%(const Decimal& left, const double right) {
  return 0.0;
}

ARIES_DEVICE double operator%(const float left, const Decimal& right) {
  return 0.0;
}

ARIES_DEVICE double operator%(const double left, const Decimal& right) {
  return 0.0;
}

// lixin 判断是否为0
ARIES_HOST_DEVICE_NO_INLINE bool Decimal::isZero() const {
  for (int i = 0; i < NUM_TOTAL_DIG; ++i) {
    if (v[i] != 0) {
      return false;
    }
  }
  return true;
}

// 10的i次方
ARIES_HOST_DEVICE_NO_INLINE int32_t Decimal::GetPowers10(int i) const {
  int32_t res = 1;
  switch (i) {
  case 0:
    res = 1;
    break;
  case 1:
    res = 10;
    break;
  case 2:
    res = 100;
    break;
  case 3:
    res = 1000;
    break;
  case 4:
    res = 10000;
    break;
  case 5:
    res = 100000;
    break;
  case 6:
    res = 1000000;
    break;
  case 7:
    res = 10000000;
    break;
  case 8:
    res = 100000000;
    break;
  case 9:
    res = 1000000000;
    break;
  default:
    break;
  }
  return res;
}

// 根据精度获取其最大值 TODO 只在检测整数部分也溢出后采用的方法 考虑只设置 error
// 位 不求该范围下的最大值 外面没有调用此函数
ARIES_HOST_DEVICE_NO_INLINE void Decimal::GenMaxDecByPrecision() {}

// 加减对齐函数
ARIES_HOST_DEVICE_NO_INLINE void Decimal::AlignAddSubData(Decimal& d) {
  if (frac == d.frac) {
    // do nothing
    return;
  }
  // 例如 a = 2.4 (frac = 1) b = 1.23 (frac = 2) ：frac < d.frac : this(a)
  // 左移一位 例如 a = 1.23 (frac = 2) b = 2.4 (frac = 1) ：frac > d.frac : b
  // 左移 一位
  if (frac < d.frac) {
    *this << d.frac - frac;
    frac = d.frac;
  } else {
    d << frac - d.frac;
    d.frac = frac;
  }
}

// 初始化精度函数
ARIES_HOST_DEVICE_NO_INLINE void Decimal::initialize(uint32_t pr, uint32_t fc,
                                                     uint32_t m) {
  // 符号 + prec
  prec = pr;
  // 小数点位数
  frac = fc;
  // 数组存储位置
  aries_memset(v, 0x00, sizeof(v));

  sign = 0;
  error = 0;
}

// 将 Decimal 转化为数字
ARIES_HOST_DEVICE_NO_INLINE double Decimal::GetDouble() const {
  double z = 0;
  for (int i = INDEX_LAST_DIG; i >= 0; i--) {
    if (v[i]) {
      z += v[i];
    }
    if (z) {
      z *= PER_DEC_MAX_SCALE;
    }
  }
  z = z / GetPowers10(frac);
  if (sign == 1) {
    z = -z;
  }
  return z;
}

// 转换为int64
ARIES_HOST_DEVICE_NO_INLINE uint64_t Decimal::ToUint64() const {
  uint64_t res = v[0] + (uint64_t) v[1] * PER_DEC_MAX_SCALE;
  return res;
}

// 无用函数
ARIES_HOST_DEVICE_NO_INLINE bool Decimal::StringToDecimal(char* str, int len) {
  return true;
}

/* mysql> select
   999999999999999999999999999999999999999999999999999999999999999999999999999999999999;
   +--------------------------------------------------------------------------------------+
   |
   999999999999999999999999999999999999999999999999999999999999999999999999999999999999
   |
   +--------------------------------------------------------------------------------------+
   | 99999999999999999999999999999999999999999999999999999999999999999 |
   +--------------------------------------------------------------------------------------+
   1 row in set, 1 warning (0.00 sec)

   mysql> show warnings;
   +---------+------+------------------------------------------------------------------------------------------------------------------------+
   | Level   | Code | Message |
   +---------+------+------------------------------------------------------------------------------------------------------------------------+
   | Warning | 1292 | Truncated incorrect DECIMAL value:
   '999999999999999999999999999999999999999999999999999999999999999999999999999999999'
   |
   +---------+------+------------------------------------------------------------------------------------------------------------------------+
*/
// string转decimal函数
ARIES_HOST_DEVICE_NO_INLINE bool Decimal::StringToDecimal(char* str) {
  memset(v, 0x00, sizeof(uint32_t) * NUM_TOTAL_DIG);
  int flag = 0;
  uint64_t temp = 0;
  uint32_t carry = 0;
  int dotflag = 0;
  int dot = 0;
  sign = 0;
  int t = 0;
  if (str[0] == '-') {
    flag = 1;
    t = 1;
  }
  for (int i = t; str[i] != '\0'; i++) {
    if (str[i] == '.') {
      dotflag = 1;
      i++;
    }
    if (dotflag != 0) {
      dot++;
    }
    // #pragma unroll
    for (int j = 0; j < NUM_TOTAL_DIG; j++) {
      temp = (uint64_t) v[j] * 10 + carry;
      v[j] = temp % MAX_INT32;
      carry = temp / MAX_INT32;
    }
    temp = (uint64_t) v[0] + str[i] - '0';
    v[0] = temp % MAX_INT32;
    carry = temp / MAX_INT32;
    // #pragma unroll
    for (int j = 1; j < NUM_TOTAL_DIG && carry > 0; j++) {
      temp = (uint64_t) v[j] + carry;
      v[j] = temp % MAX_INT32;
      carry = temp / MAX_INT32;
    }
  }
  if (flag == 1) {
    sign = 1;
  }
  frac = dot;
  return true;
}
}  // namespace aries_acc
