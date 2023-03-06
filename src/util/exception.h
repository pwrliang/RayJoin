//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#ifndef RAYJOIN_EXCEPTION_H
#define RAYJOIN_EXCEPTION_H
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <optix.h>
#include <optix_stubs.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------

#define OPTIX_CHECK(call) ::rayjoin::optixCheck(call, #call, __FILE__, __LINE__)

#define OPTIX_CHECK_LOG(call)                                                 \
  ::rayjoin::optixCheckLog(call, log, sizeof(log), sizeof_log, #call, __FILE__, \
                         __LINE__)

// This version of the log-check macro doesn't require the user do setup
// a log buffer and size variable in the surrounding context; rather the
// macro defines a log buffer and log size variable (LOG and LOG_SIZE)
// respectively that should be passed to the message being checked.
// E.g.:
//  OPTIX_CHECK_LOG2( optixProgramGroupCreate( ..., LOG, &LOG_SIZE, ... );
//
#define OPTIX_CHECK_LOG2(call)                                                \
  do {                                                                        \
    char LOG[400];                                                            \
    size_t LOG_SIZE = sizeof(LOG);                                            \
    ::rayjoin::optixCheckLog(call, LOG, sizeof(LOG), LOG_SIZE, #call, __FILE__, \
                           __LINE__);                                         \
  } while (false)

#define OPTIX_CHECK_NOTHROW(call) \
  ::rayjoin::optixCheckNoThrow(call, #call, __FILE__, __LINE__)

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK(call) ::rayjoin::cudaCheck(call, #call, __FILE__, __LINE__)

#define CUDA_SYNC_CHECK() ::rayjoin::cudaSyncCheck(__FILE__, __LINE__)

// A non-throwing variant for use in destructors.
// An iostream must be provided for output (e.g. std::cerr).
#define CUDA_CHECK_NOTHROW(call) \
  ::rayjoin::cudaCheckNoThrow(call, #call, __FILE__, __LINE__)

//------------------------------------------------------------------------------
//
// Assertions
//
//------------------------------------------------------------------------------

#define SUTIL_ASSERT(cond) \
  ::rayjoin::assertCond(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

#define SUTIL_ASSERT_MSG(cond, msg)                                     \
  ::rayjoin::assertCondMsg(static_cast<bool>(cond), #cond, msg, __FILE__, \
                         __LINE__)

#define SUTIL_ASSERT_FAIL_MSG(msg) \
  ::rayjoin::assertFailMsg(msg, __FILE__, __LINE__)

namespace rayjoin {

class Exception : public std::runtime_error {
 public:
  Exception(const char* msg) : std::runtime_error(msg) {}

  Exception(OptixResult res, const char* msg)
      : std::runtime_error(createMessage(res, msg).c_str()) {}

 private:
  std::string createMessage(OptixResult res, const char* msg) {
    std::ostringstream out;
    out << optixGetErrorName(res) << ": " << msg;
    return out.str();
  }
};

inline void optixCheck(OptixResult res, const char* call, const char* file,
                       unsigned int line) {
  if (res != OPTIX_SUCCESS) {
    std::stringstream ss;
    ss << "Optix call '" << call << "' failed: " << file << ':' << line
       << ")\n";
    throw Exception(res, ss.str().c_str());
  }
}

inline void optixCheckLog(OptixResult res, const char* log, size_t sizeof_log,
                          size_t sizeof_log_returned, const char* call,
                          const char* file, unsigned int line) {
  if (res != OPTIX_SUCCESS) {
    std::stringstream ss;
    ss << "Optix call '" << call << "' failed: " << file << ':' << line
       << ")\nLog:\n"
       << log << (sizeof_log_returned > sizeof_log ? "<TRUNCATED>" : "")
       << '\n';
    throw Exception(res, ss.str().c_str());
  }
}

inline void optixCheckNoThrow(OptixResult res, const char* call,
                              const char* file, unsigned int line) {
  if (res != OPTIX_SUCCESS) {
    std::cerr << "Optix call '" << call << "' failed: " << file << ':' << line
              << ")\n";
    std::terminate();
  }
}

inline void cudaCheck(cudaError_t error, const char* call, const char* file,
                      unsigned int line) {
  if (error != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA call (" << call << " ) failed with error: '"
       << cudaGetErrorString(error) << "' (" << file << ":" << line << ")\n";
    throw Exception(ss.str().c_str());
  }
}

inline void cudaSyncCheck(const char* file, unsigned int line) {
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream ss;
    ss << "CUDA error on synchronize with error '" << cudaGetErrorString(error)
       << "' (" << file << ":" << line << ")\n";
    throw Exception(ss.str().c_str());
  }
}

inline void cudaCheckNoThrow(cudaError_t error, const char* call,
                             const char* file, unsigned int line) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA call (" << call << " ) failed with error: '"
              << cudaGetErrorString(error) << "' (" << file << ":" << line
              << ")\n";
    std::terminate();
  }
}

inline void assertCond(bool result, const char* cond, const char* file,
                       unsigned int line) {
  if (!result) {
    std::stringstream ss;
    ss << file << " (" << line << "): " << cond;
    throw Exception(ss.str().c_str());
  }
}

inline void assertCondMsg(bool result, const char* cond, const std::string& msg,
                          const char* file, unsigned int line) {
  if (!result) {
    std::stringstream ss;
    ss << msg << ": " << file << " (" << line << "): " << cond;
    throw Exception(ss.str().c_str());
  }
}

[[noreturn]] inline void assertFailMsg(const std::string& msg, const char* file,
                                       unsigned int line) {
  std::stringstream ss;
  ss << msg << ": " << file << " (" << line << ')';
  throw Exception(ss.str().c_str());
}

}

#endif  // RAYJOIN_EXCEPTION_H
