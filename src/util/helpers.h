

#ifndef UTIL_HELPERS_H
#define UTIL_HELPERS_H

#include <cuda_runtime.h>
#include <optix_device.h>
#include <vector_types.h>

#include "util/vec_math.h"
/**
 * Convert double to float with rounding
 *
 * @v double
 * @dir dir = -1, round down; dir = 1 round up
 * @iter number of calling nextafter
 */
__forceinline__ __device__ __host__ float next_float_from_double(double v,
                                                                 int dir,
                                                                 int iter = 1) {
  assert(dir == 1 || dir == -1);
  auto fv = static_cast<float>(v);  // pos number
  if (fv == 0) {
    return 0.0f;
  }
  float to = v * dir < 0 ? 0 : dir * std::numeric_limits<float>::infinity();

  for (int i = 0; i < iter; i++) {
    fv = std::nextafter(fv, to);
  }

  return fv;
};

__forceinline__ __device__ __host__ void print_aabb(const OptixAabb& aabb) {
  printf("x: [%f,%f], y: [%f, %f], z: [%f, %f]\n", aabb.minX, aabb.maxX,
         aabb.minY, aabb.maxY, aabb.minZ, aabb.maxZ);
}

__forceinline__ __device__ bool operator>(const float3 a, const float3 b) {
  return (a.x > b.x && a.y > b.y && a.z > b.z);
}

__forceinline__ __device__ bool operator<(const float3 a, const float3 b) {
  return (a.x < b.x && a.y < b.y && a.z < b.z);
}

__forceinline__ __device__ bool operator>=(const float3 a, const float3 b) {
  return (a.x >= b.x && a.y >= b.y && a.z >= b.z);
}

__forceinline__ __device__ bool operator<=(const float3 a, const float3 b) {
  return (a.x <= b.x && a.y <= b.y && a.z <= b.z);
}

__forceinline__ __device__ bool operator==(const float3 a, const float3 b) {
  return (a.x == b.x && a.y == b.y && a.z == b.z);
}

__forceinline__ __device__ void* unpackPointer(unsigned int i0,
                                               unsigned int i1) {
  const unsigned long long uptr =
      static_cast<unsigned long long>(i0) << 32 | i1;
  void* ptr = reinterpret_cast<void*>(uptr);
  return ptr;
}

__forceinline__ __device__ void packPointer(void* ptr, unsigned int& i0,
                                            unsigned int& i1) {
  const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

template <typename T>
__forceinline__ __device__ void unpack64(unsigned int i0, unsigned int i1,
                                         T* t) {
  assert(sizeof(T) == 8);
  *reinterpret_cast<unsigned long long*>(t) =
      static_cast<unsigned long long>(i0) << 32 | i1;
}

template <typename T>
__forceinline__ __device__ void pack64(T* t, unsigned int& i0,
                                       unsigned int& i1) {
  assert(sizeof(T) == 8);
  const unsigned long long ud = *reinterpret_cast<unsigned long long*>(t);
  i0 = ud >> 32;
  i1 = ud & 0x00000000ffffffff;
}

template <typename T>
__forceinline__ __device__ T* getPRD() {
  const unsigned int u0 = optixGetPayload_0();
  const unsigned int u1 = optixGetPayload_1();
  return reinterpret_cast<T*>(unpackPointer(u0, u1));
}

__forceinline__ __device__ float luminance(const float3& rgb) {
  const float3 ntsc_luminance = {0.30f, 0.59f, 0.11f};
  return dot(rgb, ntsc_luminance);
}

__forceinline__ __device__ float fresnel_schlick(const float cos_theta,
                                                 const float exponent = 5.0f,
                                                 const float minimum = 0.0f,
                                                 const float maximum = 1.0f) {
  /**
    Clamp the result of the arithmetic due to floating point precision:
    the result should lie strictly within [minimum, maximum]
    return clamp(minimum + (maximum - minimum) * powf(1.0f - cos_theta,
    exponent), minimum, maximum);
  */

  /** The max doesn't seem like it should be necessary, but without it you get
      annoying broken pixels at the center of reflective spheres where cos_theta
     ~ 1.
  */
  return clamp(minimum + (maximum - minimum) *
                             powf(fmaxf(0.0f, 1.0f - cos_theta), exponent),
               minimum, maximum);
}

__forceinline__ __device__ float3 fresnel_schlick(const float cos_theta,
                                                  const float exponent,
                                                  const float3& minimum,
                                                  const float3& maximum) {
  return make_float3(
      fresnel_schlick(cos_theta, exponent, minimum.x, maximum.x),
      fresnel_schlick(cos_theta, exponent, minimum.y, maximum.y),
      fresnel_schlick(cos_theta, exponent, minimum.z, maximum.z));
}

__forceinline__ __device__ bool refract(float3& r, const float3& i,
                                        const float3& n, const float ior) {
  float3 nn = n;
  float negNdotV = dot(i, nn);
  float eta;

  if (negNdotV > 0.0f) {
    eta = ior;
    nn = -n;
    negNdotV = -negNdotV;
  } else {
    eta = 1.f / ior;
  }

  const float k = 1.f - eta * eta * (1.f - negNdotV * negNdotV);

  if (k < 0.0f) {
    // Initialize this value, so that r always leaves this function initialized.
    r = make_float3(0.f);
    return false;
  } else {
    r = normalize(eta * i - (eta * negNdotV + sqrtf(k)) * nn);
    return true;
  }
}

__forceinline__ __device__ float3 exp(const float3& x) {
  return make_float3(exp(x.x), exp(x.y), exp(x.z));
}

#define float3_as_args(u)                     \
  reinterpret_cast<unsigned int&>((u).x),     \
      reinterpret_cast<unsigned int&>((u).y), \
      reinterpret_cast<unsigned int&>((u).z)

#endif  // UTIL_HELPERS_H
