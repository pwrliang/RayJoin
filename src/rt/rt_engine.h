#ifndef RAYJOIN_RT_RT_ENGINE
#define RAYJOIN_RT_RT_ENGINE
#include <cuda.h>
#include <optix_types.h>
#include <thrust/device_vector.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "algo/launch_parameters.h"
#include "glog/logging.h"
#include "rt/sbt_record.h"
#include "util/queue.h"
#include "util/shared_array.h"
#include "util/shared_value.h"
#include "util/stream.h"
#define MODULE_ENABLE_MISS (1 << 0)
#define MODULE_ENABLE_CH (1 << 1)
#define MODULE_ENABLE_AH (1 << 2)
#define MODULE_ENABLE_IS (1 << 3)

namespace rayjoin {

enum ModuleIdentifier {
  MODULE_ID_EXTERNAL,
  MODULE_ID_LSI,
  MODULE_ID_LSI_CUSTOM,
  MODULE_ID_PIP,
  MODULE_ID_PIP_CUSTOM,
  NUM_MODULE_IDENTIFIERS
};

class Module {
 public:
  Module() = default;

  Module(ModuleIdentifier id)
      : id_(id), enabled_module_(0), n_payload_(0), n_attribute_(0) {}

  void EnableMiss() { enabled_module_ |= MODULE_ENABLE_MISS; }

  void EnableClosestHit() { enabled_module_ |= MODULE_ENABLE_CH; }

  void EnableAnyHit() { enabled_module_ |= MODULE_ENABLE_AH; }

  void EnableIsIntersection() { enabled_module_ |= MODULE_ENABLE_IS; }

  bool IsMissEnable() const { return enabled_module_ & MODULE_ENABLE_MISS; }

  bool IsClosestHitEnable() const { return enabled_module_ & MODULE_ENABLE_CH; }

  bool IsAnyHitEnable() const { return enabled_module_ & MODULE_ENABLE_AH; }

  bool IsIsIntersectionEnabled() const {
    return enabled_module_ & MODULE_ENABLE_IS;
  }

  void set_program_name(const std::string& program_name) {
    program_name_ = program_name;
  }
  const std::string& get_program_name() const { return program_name_; }

  void set_function_suffix(const std::string& function_suffix) {
    function_suffix_ = function_suffix;
  }
  const std::string& get_function_suffix() const { return function_suffix_; }

  void set_launch_params_name(const std::string& launch_params_name) {
    launch_params_name_ = launch_params_name;
  }

  const std::string& get_launch_params_name() const {
    return launch_params_name_;
  }

  void set_n_payload(int n_payload) { n_payload_ = n_payload; }

  int get_n_payload() const { return n_payload_; }

  void set_n_attribute(int n_attribute) { n_attribute_ = n_attribute; }

  int get_n_attribute() const { return n_attribute_; }

  ModuleIdentifier get_id() const { return id_; }

 private:
  ModuleIdentifier id_;
  std::string program_name_;
  std::string function_suffix_;
  std::string launch_params_name_;
  int enabled_module_;

  int n_payload_;
  int n_attribute_;
};

struct RTConfig {
  RTConfig()
      : max_reg_count(0),
        max_trace_depth(2),
        opt_level(OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        dbg_level(OPTIX_COMPILE_DEBUG_LEVEL_NONE),
        temp_buf_size(100 * 1024 * 1024),
        output_buf_size(500 * 1024 * 1024) {}

  void AddModule(const Module& mod) { modules[mod.get_id()] = mod; }

  int max_reg_count;
  int max_trace_depth;
  OptixCompileOptimizationLevel opt_level;
  OptixCompileDebugLevel dbg_level;
  std::map<ModuleIdentifier, Module> modules;
  size_t temp_buf_size;
  size_t output_buf_size;
};

RTConfig get_default_rt_config(const std::string& exec_root);

class RTEngine {
 public:
  RTEngine() = default;

  void Init(const RTConfig& config) {
    initOptix(config);
    createContext();
    createModule(config);
    createExternalPrograms();
    createRaygenPrograms(config);
    createMissPrograms(config);
    createHitgroupPrograms(config);
    createPipeline(config);
    buildSBT(config);
  }

  OptixTraversableHandle BuildAccelTriangles(Stream& stream,
                                             ArrayView<float3> vertices,
                                             ArrayView<uint3> indices = {
                                                 nullptr, 0}) {
    return buildAccelTriangle(stream, vertices, indices);
  }

  OptixTraversableHandle BuildAccelCustom(Stream& stream,
                                          ArrayView<OptixAabb> aabbs) {
    return buildAccel(stream, aabbs);
  }

  void FreeBVH(OptixTraversableHandle handle);

  void Render(Stream& stream, ModuleIdentifier mod, dim3 dim);

  template <typename T>
  void CopyLaunchParams(Stream& stream, const T& params) {
    auto* begin = reinterpret_cast<const char*>(&params);

    h_launch_params_.assign(begin, begin + sizeof(params));
    launch_params_.resize(h_launch_params_.size());

    LOG(INFO) << "Parm size: " << launch_params_.size();
    thrust::copy(thrust::cuda::par.on(stream.cuda_stream()),
                 h_launch_params_.begin(), h_launch_params_.end(),
                 launch_params_.begin());
  }

 protected:
  void initOptix(const RTConfig& config);

  void createContext();

  void createModule(const RTConfig& config);

  void createExternalPrograms();

  void createRaygenPrograms(const RTConfig& config);

  void createMissPrograms(const RTConfig& config);

  void createHitgroupPrograms(const RTConfig& config);

  void createPipeline(const RTConfig& config);

  void buildSBT(const RTConfig& config);

  OptixTraversableHandle buildAccelTriangle(Stream& stream,
                                            ArrayView<float3> vertices,
                                            ArrayView<uint3> indices);

  OptixTraversableHandle buildAccel(Stream& stream, ArrayView<OptixAabb> aabbs);

  std::vector<char> readData(const std::string& filename) {
    std::ifstream inputData(filename, std::ios::binary);

    if (inputData.fail()) {
      std::cerr << "ERROR: readData() Failed to open file " << filename << '\n';
      return {};
    }

    // Copy the input buffer to a char vector.
    std::vector<char> data(std::istreambuf_iterator<char>(inputData), {});

    if (inputData.fail()) {
      std::cerr << "ERROR: readData() Failed to read file " << filename << '\n';
      return {};
    }

    return data;
  }

  CUcontext cuda_context_;
  OptixDeviceContext optix_context_;

  // modules that contains device program
  std::vector<OptixModule> modules_;
  OptixModuleCompileOptions module_compile_options_ = {};

  std::vector<OptixPipeline> pipelines_;
  std::vector<OptixPipelineCompileOptions> pipeline_compile_options_;
  OptixPipelineLinkOptions pipeline_link_options_ = {};

  std::vector<OptixProgramGroup> external_pgs_;

  std::vector<OptixProgramGroup> raygen_pgs_;
  std::vector<thrust::device_vector<RaygenRecord>> raygen_records_;

  std::vector<OptixProgramGroup> miss_pgs_;
  std::vector<thrust::device_vector<MissRecord>> miss_records_;

  std::vector<OptixProgramGroup> hitgroup_pgs_;
  std::vector<thrust::device_vector<HitgroupRecord>> hitgroup_records_;
  std::vector<OptixShaderBindingTable> sbts_;

  // device data
  thrust::device_vector<unsigned char> temp_buf_, output_buf_;
  std::map<OptixTraversableHandle,
           std::unique_ptr<thrust::device_vector<unsigned char>>>
      as_buffers_;

  thrust::device_vector<char> h_launch_params_;
  thrust::device_vector<char> launch_params_;
};

}  // namespace rayjoin

#endif  // RAYJOIN_RT_RT_ENGINE
