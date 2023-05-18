#include <cuda.h>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>  // for g_optixFunctionTable
#include <optix_host.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <iostream>
#include <stdexcept>

#include "glog/logging.h"
#include "rt/rt_engine.h"
#include "rt/sbt_record.h"
#include "util/exception.h"
#include "util/markers.h"
#include "util/stopwatch.h"
#include "util/stream.h"
#include "util/util.h"

namespace rayjoin {
RTConfig get_default_rt_config(const std::string& exec_root) {
  RTConfig config;

  Module mod_lsi_custom(ModuleIdentifier::MODULE_ID_LSI_CUSTOM);

  mod_lsi_custom.set_program_name(exec_root + "/ptx/rt_lsi_custom.ptx");
  mod_lsi_custom.set_function_suffix("lsi");
  mod_lsi_custom.set_launch_params_name("params");
  mod_lsi_custom.EnableIsIntersection();
  mod_lsi_custom.set_n_payload(1);
  if (access(mod_lsi_custom.get_program_name().c_str(), R_OK) != 0) {
    LOG(FATAL) << "Cannot open " << mod_lsi_custom.get_program_name();
  }
  config.AddModule(mod_lsi_custom);

  Module mod_pip_custom(ModuleIdentifier::MODULE_ID_PIP_CUSTOM);

  mod_pip_custom.set_program_name(exec_root + "/ptx/rt_pip_custom.ptx");
  mod_pip_custom.set_function_suffix("pip_custom");
  mod_pip_custom.set_launch_params_name("params");
  mod_pip_custom.EnableIsIntersection();
  mod_pip_custom.set_n_payload(4);

  if (access(mod_pip_custom.get_program_name().c_str(), R_OK) != 0) {
    LOG(FATAL) << "Cannot open " << mod_pip_custom.get_program_name();
  }
  config.AddModule(mod_pip_custom);

#ifndef NDEBUG
  config.opt_level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  config.dbg_level = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
  config.opt_level = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
  config.dbg_level = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

  return config;
}

extern "C" char embedded_ptx_code_overlay[];
void RTEngine::initOptix(const RTConfig& config) {
  // https://stackoverflow.com/questions/10415204/how-to-create-a-cuda-context
  cudaFree(0);
  int numDevices;
  cudaGetDeviceCount(&numDevices);
  if (numDevices == 0)
    throw std::runtime_error("#osc: no CUDA capable devices found!");

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK(optixInit());
  temp_buf_.reserve(config.temp_buf_size);
  output_buf_.reserve(config.output_buf_size);
}

static void context_log_cb(unsigned int level, const char* tag,
                           const char* message, void*) {
  fprintf(stderr, "[%2d][%12s]: %s\n", (int) level, tag, message);
}

void RTEngine::createContext() {
  CUresult cu_res = cuCtxGetCurrent(&cuda_context_);
  if (cu_res != CUDA_SUCCESS)
    fprintf(stderr, "Error querying current context: error code %d\n", cu_res);
  OptixDeviceContextOptions options;
  options.logCallbackFunction = context_log_cb;
  options.logCallbackData = nullptr;

#ifndef NDEBUG
  options.logCallbackLevel = 4;
  options.validationMode = OptixDeviceContextValidationMode::
      OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
  options.logCallbackLevel = 2;
#endif
  OPTIX_CHECK(
      optixDeviceContextCreate(cuda_context_, &options, &optix_context_));
}

void RTEngine::createModule(const RTConfig& config) {
  module_compile_options_.maxRegisterCount = config.max_reg_count;
  module_compile_options_.optLevel = config.opt_level;
  module_compile_options_.debugLevel = config.dbg_level;
  pipeline_compile_options_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

  pipeline_link_options_.maxTraceDepth = config.max_trace_depth;

  auto& conf_modules = config.modules;

  modules_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

  for (auto& pair : conf_modules) {
    std::vector<char> programData = readData(pair.second.get_program_name());
    auto& pipeline_compile_options = pipeline_compile_options_[pair.first];

    pipeline_compile_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.numPayloadValues = pair.second.get_n_payload();
    pipeline_compile_options.numAttributeValues = pair.second.get_n_attribute();
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName =
        pair.second.get_launch_params_name().c_str();

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(optix_context_, &module_compile_options_,
                                  &pipeline_compile_options, programData.data(),
                                  programData.size(), log, &sizeof_log,
                                  &modules_[pair.first]));
    if (sizeof_log > 1) {
      VLOG(1) << log;
    }
  }
}

void RTEngine::createExternalPrograms() {
  //  external_pgs_.resize(1);
  //
  //  OptixProgramGroupDesc pgd;
  //  OptixProgramGroupOptions pgOptions = {};
  //
  //  pgd.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
  //  pgd.flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
  //  pgd.callables.moduleDC = modules_[MODULE_ID_EXTERNAL];
  //  pgd.callables.entryFunctionNameDC = "__direct_callable__dummy_func";
  //  pgd.callables.moduleCC = nullptr;
  //  pgd.callables.entryFunctionNameCC = nullptr;
  //
  //  char log[2048];
  //  size_t sizeof_log = sizeof(log);
  //  OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pgd, 1, &pgOptions,
  //  log,
  //                                      &sizeof_log, &external_pgs_[0]));
  //  if (sizeof_log > 1) {
  //    std::cout << log << std::endl;
  //  }
}

void RTEngine::createRaygenPrograms(const RTConfig& config) {
  const auto& conf_modules = config.modules;
  raygen_pgs_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

  for (auto& pair : conf_modules) {
    auto f_name = "__raygen__" + pair.second.get_function_suffix();
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = modules_[pair.first];
    pgDesc.raygen.entryFunctionName = f_name.c_str();

    // OptixProgramGroup raypg;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pgDesc, 1, &pgOptions,
                                        log, &sizeof_log,
                                        &raygen_pgs_[pair.first]));
    if (sizeof_log > 1) {
      LOG(ERROR) << log;
    }
  }
}

/*! does all setup for the miss program(s) we are going to use */
void RTEngine::createMissPrograms(const RTConfig& config) {
  const auto& conf_modules = config.modules;
  miss_pgs_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

  for (auto& pair : conf_modules) {
    auto& mod = pair.second;
    auto f_name = "__miss__" + mod.get_function_suffix();
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

    pgDesc.miss.module = nullptr;
    pgDesc.miss.entryFunctionName = nullptr;

    if (mod.IsMissEnable()) {
      pgDesc.miss.module = modules_[pair.first];
      pgDesc.miss.entryFunctionName = f_name.c_str();
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pgDesc, 1, &pgOptions,
                                        log, &sizeof_log,
                                        &miss_pgs_[pair.first]));
    if (sizeof_log > 1) {
      VLOG(1) << log;
    }
  }
}

/*! does all setup for the hitgroup program(s) we are going to use */
void RTEngine::createHitgroupPrograms(const RTConfig& config) {
  auto& conf_modules = config.modules;
  hitgroup_pgs_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

  for (auto& pair : conf_modules) {
    const auto& conf_mod = pair.second;
    auto f_name_anythit = "__anyhit__" + conf_mod.get_function_suffix();
    auto f_name_intersect = "__intersection__" + conf_mod.get_function_suffix();
    auto f_name_closesthit = "__closesthit__" + conf_mod.get_function_suffix();
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pg_desc = {};

    pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    pg_desc.hitgroup.moduleIS = nullptr;
    pg_desc.hitgroup.entryFunctionNameIS = nullptr;
    pg_desc.hitgroup.moduleAH = nullptr;
    pg_desc.hitgroup.entryFunctionNameAH = nullptr;
    pg_desc.hitgroup.moduleCH = nullptr;
    pg_desc.hitgroup.entryFunctionNameCH = nullptr;

    if (conf_mod.IsIsIntersectionEnabled()) {
      pg_desc.hitgroup.moduleIS = modules_[pair.first];
      pg_desc.hitgroup.entryFunctionNameIS = f_name_intersect.c_str();
    }

    if (conf_mod.IsAnyHitEnable()) {
      pg_desc.hitgroup.moduleAH = modules_[pair.first];
      pg_desc.hitgroup.entryFunctionNameAH = f_name_anythit.c_str();
    }

    if (conf_mod.IsClosestHitEnable()) {
      pg_desc.hitgroup.moduleCH = modules_[pair.first];
      pg_desc.hitgroup.entryFunctionNameCH = f_name_closesthit.c_str();
    }

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &pg_desc, 1, &pgOptions,
                                        log, &sizeof_log,
                                        &hitgroup_pgs_[pair.first]));
    if (sizeof_log > 1) {
      VLOG(1) << log;
    }
  }
}

/*! assembles the full pipeline of all programs */
void RTEngine::createPipeline(const RTConfig& config) {
  pipelines_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

  for (auto& pair : config.modules) {
    std::vector<OptixProgramGroup> program_groups;
    program_groups.push_back(raygen_pgs_[pair.first]);
    program_groups.push_back(miss_pgs_[pair.first]);
    program_groups.push_back(hitgroup_pgs_[pair.first]);

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        optix_context_, &pipeline_compile_options_[pair.first],
        &pipeline_link_options_, program_groups.data(),
        (int) program_groups.size(), log, &sizeof_log,
        &pipelines_[pair.first]));
    if (sizeof_log > 1) {
      VLOG(1) << log;
    }

    OptixStackSizes stack_sizes = {};
    for (auto& prog_group : program_groups) {
      OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes,
                                                pipelines_[pair.first]));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, config.max_trace_depth,
        0,  // maxCCDepth
        0,  // maxDCDepth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipelines_[pair.first], direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        1  // maxTraversableDepth
        ));
  }
}

/*! constructs the shader binding table */
void RTEngine::buildSBT(const RTConfig& config) {
  sbts_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);
  raygen_records_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);
  miss_records_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);
  hitgroup_records_.resize(ModuleIdentifier::NUM_MODULE_IDENTIFIERS);

  for (auto& pair : config.modules) {
    auto& sbt = sbts_[pair.first];
    std::vector<RaygenRecord> raygenRecords;
    {
      RaygenRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pgs_[pair.first], &rec));
      rec.data = nullptr; /* for now ... */
      raygenRecords.push_back(rec);
    }
    raygen_records_[pair.first] = raygenRecords;
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(
        thrust::raw_pointer_cast(raygen_records_[pair.first].data()));

    std::vector<MissRecord> missRecords;
    {
      MissRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(miss_pgs_[pair.first], &rec));
      rec.data = nullptr; /* for now ... */
      missRecords.push_back(rec);
    }

    miss_records_[pair.first] = missRecords;
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(
        thrust::raw_pointer_cast(miss_records_[pair.first].data()));
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount = (int) missRecords.size();

    std::vector<HitgroupRecord> hitgroupRecords;
    {
      HitgroupRecord rec;
      OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pgs_[pair.first], &rec));
      rec.data = nullptr;
      hitgroupRecords.push_back(rec);
    }
    hitgroup_records_[pair.first] = hitgroupRecords;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(
        thrust::raw_pointer_cast(hitgroup_records_[pair.first].data()));
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount = (int) hitgroupRecords.size();
  }
}

OptixTraversableHandle RTEngine::buildAccel(Stream& stream,
                                            ArrayView<OptixAabb> aabbs) {
  OptixTraversableHandle traversable;
  OptixBuildInput build_input = {};
  CUdeviceptr d_aabb = THRUST_TO_CUPTR(aabbs.data());
  // Setup AABB build input. Don't disable AH.
  // OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
  uint32_t build_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
  uint32_t num_prims = aabbs.size();

  CHECK_EQ(reinterpret_cast<uint64_t>(aabbs.data()) %
               OPTIX_AABB_BUFFER_BYTE_ALIGNMENT,
           0);

  build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
  build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
  build_input.customPrimitiveArray.flags = build_input_flags;
  build_input.customPrimitiveArray.numSbtRecords = 1;
  build_input.customPrimitiveArray.numPrimitives = num_prims;
  // it's important to pass 0 to sbtIndexOffsetBuffer
  build_input.customPrimitiveArray.sbtIndexOffsetBuffer = 0;
  build_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
  build_input.customPrimitiveArray.primitiveIndexOffset = 0;

  // ==================================================================
  // Bottom-level acceleration structure (BLAS) setup
  // ==================================================================

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags =
      OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
  accelOptions.motionOptions.numKeys = 1;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions,
                                           &build_input,
                                           1,  // num_build_inputs
                                           &blas_buffer_sizes));

  VLOG(1) << "Building AS, num prims: " << num_prims
          << ", Required Temp Size: " << blas_buffer_sizes.tempSizeInBytes
          << " Output Size: " << blas_buffer_sizes.outputSizeInBytes;

  // ==================================================================
  // prepare compaction
  // ==================================================================

  SharedValue<uint64_t> compacted_size;

  OptixAccelEmitDesc emitDesc;
  emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = reinterpret_cast<CUdeviceptr>(compacted_size.data());

  // ==================================================================
  // execute build (main stage)
  // ==================================================================
  {
    IntervalRangeMarker marker_alloc(
        blas_buffer_sizes.tempSizeInBytes + blas_buffer_sizes.outputSizeInBytes,
        "Allocate blas buffer");
    temp_buf_.resize(blas_buffer_sizes.tempSizeInBytes);
    output_buf_.resize(blas_buffer_sizes.outputSizeInBytes);
  }
  {
    RangeMarker marker(true, "AccelBuild");
    OPTIX_CHECK(
        optixAccelBuild(optix_context_, stream.cuda_stream(), &accelOptions,
                        &build_input, 1, THRUST_TO_CUPTR(temp_buf_.data()),
                        temp_buf_.size(), THRUST_TO_CUPTR(output_buf_.data()),
                        output_buf_.size(), &traversable, &emitDesc, 1));
  }
  // ==================================================================
  // perform compaction
  // ==================================================================
  auto as_buffer = std::make_unique<thrust::device_vector<unsigned char>>(
      compacted_size.get(stream));
  {
    RangeMarker marker(true, "AccelCompact");
    OPTIX_CHECK(optixAccelCompact(
        optix_context_, stream.cuda_stream(), traversable,
        THRUST_TO_CUPTR(as_buffer->data()), as_buffer->size(), &traversable));
  }
  stream.Sync();
  as_buffers_[traversable] = std::move(as_buffer);
  return traversable;
}

OptixTraversableHandle RTEngine::buildAccelTriangle(Stream& stream,
                                                    ArrayView<float3> vertices,
                                                    ArrayView<uint3> indices) {
  OptixTraversableHandle traversable;
  OptixBuildInput build_input = {};
  auto d_indices = reinterpret_cast<CUdeviceptr>(indices.data());
  auto d_vertices = reinterpret_cast<CUdeviceptr>(vertices.data());
  // Setup AABB build input. Don't disable AH.
  uint32_t build_input_flags[1] = {
      OPTIX_GEOMETRY_FLAG_NONE |
      OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};

  build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
  build_input.triangleArray.numVertices = vertices.size();
  build_input.triangleArray.vertexBuffers = &d_vertices;

  if (indices.empty()) {
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
  } else {
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.numIndexTriplets = indices.size();
    build_input.triangleArray.indexBuffer = d_indices;
  }
  build_input.triangleArray.flags = build_input_flags;
  build_input.triangleArray.numSbtRecords = 1;
  // it's important to pass 0 to sbtIndexOffsetBuffer
  build_input.triangleArray.sbtIndexOffsetBuffer = 0;
  build_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
  build_input.triangleArray.primitiveIndexOffset = 0;

  // ==================================================================
  // Bottom-level acceleration structure (BLAS) setup
  // ==================================================================

  OptixAccelBuildOptions accelOptions = {};
  accelOptions.buildFlags =
      OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
#ifndef NDEBUG
  accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
#endif
  accelOptions.motionOptions.numKeys = 1;
  accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes blas_buffer_sizes;
  OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accelOptions,
                                           &build_input,
                                           1,  // num_build_inputs
                                           &blas_buffer_sizes));

  LOG(INFO) << "Building AS, num vertices: " << vertices.size()
            << ", Required Temp Size: " << blas_buffer_sizes.tempSizeInBytes
            << " Output Size: " << blas_buffer_sizes.outputSizeInBytes;

  // ==================================================================
  // prepare compaction
  // ==================================================================

  SharedValue<uint64_t> compacted_size;

  OptixAccelEmitDesc emitDesc;
  emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitDesc.result = reinterpret_cast<CUdeviceptr>(compacted_size.data());

  // ==================================================================
  // execute build (main stage)
  // ==================================================================

  temp_buf_.resize(blas_buffer_sizes.tempSizeInBytes);
  output_buf_.resize(blas_buffer_sizes.outputSizeInBytes);

  OPTIX_CHECK(
      optixAccelBuild(optix_context_, stream.cuda_stream(), &accelOptions,
                      &build_input, 1, THRUST_TO_CUPTR(temp_buf_.data()),
                      temp_buf_.size(), THRUST_TO_CUPTR(output_buf_.data()),
                      output_buf_.size(), &traversable, &emitDesc, 1));
  stream.Sync();
  // ==================================================================
  // perform compaction
  // ==================================================================
  auto as_buffer = std::make_unique<thrust::device_vector<unsigned char>>(
      compacted_size.get(stream));

  OPTIX_CHECK(optixAccelCompact(optix_context_, stream.cuda_stream(),
                                traversable, THRUST_TO_CUPTR(as_buffer->data()),
                                as_buffer->size(), &traversable));
  stream.Sync();

  as_buffers_[traversable] = std::move(as_buffer);
  return traversable;
}

void RTEngine::FreeBVH(OptixTraversableHandle handle) {
  auto it = as_buffers_.find(handle);

  if (it != as_buffers_.end()) {
    it->second.reset(nullptr);
    as_buffers_.erase(it);

    output_buf_.resize(0);
    output_buf_.shrink_to_fit();
    temp_buf_.resize(0);
    temp_buf_.shrink_to_fit();
  }
}

void RTEngine::Render(Stream& stream, ModuleIdentifier mod, dim3 dim) {
  LOG(INFO) << "optixLaunch, [w,h,d] = " << dim.x << "," << dim.y << ","
            << dim.z;
  void* launch_params = thrust::raw_pointer_cast(launch_params_.data());
  size_t launch_params_size = launch_params_.size();

  OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                          pipelines_[mod], stream.cuda_stream(),
                          /*! parameters and SBT */
                          reinterpret_cast<CUdeviceptr>(launch_params),
                          launch_params_size, &sbts_[mod],
                          /*! dimensions of the launch: */
                          dim.x, dim.y, dim.z));
}

}  // namespace rayjoin