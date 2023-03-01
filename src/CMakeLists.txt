add_definitions(-DLEN_fir)

include_directories(${OptiX_INCLUDE})
include_directories(${PROJECT_SOURCE_DIR}/deps/Declib)
include_directories(${PROJECT_SOURCE_DIR}/deps/lbvh)

include_directories(${PROJECT_SOURCE_DIR}/src)

set(OPTIX_MODULE_EXTENSION ".ptx")
set(OPTIX_PROGRAM_TARGET "--ptx")
set(ENABLED_ARCHS "86")

set(SHADERS
        algo/rt_pip.cu
        algo/rt_lsi.cu
        algo/rt_lsi_custom.cu
        algo/rt_pip_custom.cu
        ${PROJECT_SOURCE_DIR}/deps/Declib/decimal.cu
        )

NVCUDA_COMPILE_MODULE(
        SOURCES ${SHADERS}
        DEPENDENCIES ${SHADERS_HEADERS}
        TARGET_PATH "${MODULE_TARGET_DIR}/ptx"
        EXTENSION "${OPTIX_MODULE_EXTENSION}"
        GENERATED_FILES PROGRAM_MODULES
        NVCC_OPTIONS "${OPTIX_PROGRAM_TARGET}"
#        "--gpu-architecture=compute_${ENABLED_ARCHS}"
        "--use_fast_math"
        "--relocatable-device-code=true"
        "--expt-relaxed-constexpr"
        "-Wno-deprecated-gpu-targets"
        "-I${OptiX_INCLUDE}"
        "-I${CMAKE_CURRENT_SOURCE_DIR}"
        "-I${PROJECT_SOURCE_DIR}/deps/Declib" "-DLEN_fir"
)

message("PROGRAM_MODULES: ${PROGRAM_MODULES}")



add_executable(polyover_exec
        overlay.cc
        run_overlay.cu
        rt/rt_engine.cu
        ${PROJECT_SOURCE_DIR}/deps/Declib/decimal.cu
        ${PROGRAM_MODULES}
        flags.cc
        )

target_link_libraries(polyover_exec
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_nvToolsExt_LIBRARY}
        ${GFLAGS_LIBRARIES}
        ${GLOG_LIBRARIES}
        )
target_compile_options(polyover_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
set_target_properties(polyover_exec PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON) # 70;

add_executable(query_exec
        query.cc
        run_query.cu
        rt/rt_engine.cu
        ${PROGRAM_MODULES}
        flags.cc
        )

target_link_libraries(query_exec
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        ${CUDA_nvToolsExt_LIBRARY}
        ${GFLAGS_LIBRARIES}
        ${GLOG_LIBRARIES}
        )
target_compile_options(query_exec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --use_fast_math>)
set_target_properties(query_exec PROPERTIES CUDA_ARCHITECTURES "${ENABLED_ARCHS}" CUDA_SEPARABLE_COMPILATION ON) # 70;
