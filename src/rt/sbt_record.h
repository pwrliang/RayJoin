#ifndef OPTIX_SORT_SBT_RECORD_H
#define OPTIX_SORT_SBT_RECORD_H
#include <optix_types.h>
namespace rayjoin {
/*! SBT record for a raygen program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void* data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void* data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void* data;
};

}  // namespace rayjoin

#endif  // OPTIX_SORT_SBT_RECORD_H
