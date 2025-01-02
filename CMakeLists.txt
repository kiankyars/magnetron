# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

cmake_minimum_required(VERSION 3.18)

project(magnetron LANGUAGES C)

message("Configuring magnetron project for ${CMAKE_SYSTEM_PROCESSOR}...")

set(CMAKE_C_STANDARD 99)
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

option(MAGNETRON_BUILD_TESTS "Build tests" ON)
option(MAGNETRON_BUILD_BENCHMARKS "Build benchmarks" ON)
option(MAGNETRON_BUILD_FUZZERS "Build fuzzers" OFF)

# (CPU only) Enable SIMD math function approximations. Greatly increases performance. Try disabling if you encounter numerical instability. Does NOT enable -ffast-math or similar compiler flags.
option(MAGNETRON_CPU_APPROX_MATH "Use faster but less precise math functions" ON)
option(MAGNETRON_ENABLE_CUDA "Enable CUDA support" ON) # Enable CUDA support
if (APPLE)
    option(MAGNETRON_ENABLE_ACCELERATE "Use Apple's Accelerate framework" ON)
endif()

if (${MAGNETRON_BUILD_TESTS})
    enable_testing()
endif()

set(MAGNETRON_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc" CACHE STRING "Path to the CUDA compiler") # Set to your CUDA compiler path
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(IS_AMD64 TRUE)
else()
    set(IS_AMD64 FALSE)
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
    set(IS_ARM64 TRUE)
else()
    set(IS_ARM64 FALSE)
endif()

file(GLOB_RECURSE MAGNETRON_SOURCES magnetron/*.h magnetron/*.c magnetron/*.m)
add_library(magnetron SHARED ${MAGNETRON_SOURCES})
target_include_directories(magnetron PRIVATE extern)
if(WIN32) # Windows specific config
    target_compile_options(magnetron PRIVATE /W3 /Oi)
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations for release builds
        target_compile_options(magnetron PRIVATE /O2 /Oy)
    endif()
else() # POSIX specific config
    target_link_libraries(magnetron m) # link math library
    target_compile_options(magnetron PRIVATE
        -Wall
        -Werror
        -std=c99
        -Wno-gnu-zero-variadic-macro-arguments
        -Wno-error=overflow
        -Wno-error=unused-function
        -fvisibility=hidden
        -std=gnu99
    )
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(magnetron PRIVATE -Wno-error=format-truncation)
    endif()
    if (APPLE) # Apple specific configuration
        if (${MAGNETRON_ENABLE_ACCELERATE})
            find_library(ACCELERATE_FRAMEWORK Accelerate)
            if (ACCELERATE_FRAMEWORK)
                message(STATUS "Accelerate framework found")
                target_compile_definitions(magnetron PRIVATE MAG_ACCELERATE)
                target_compile_definitions(magnetron PRIVATE ACCELERATE_NEW_LAPACK)
                target_compile_definitions(magnetron PRIVATE ACCELERATE_LAPACK_ILP64)
                target_link_libraries(magnetron ${ACCELERATE_FRAMEWORK})
            else()
                message(WARNING "Accelerate framework not found, using fallback")
            endif()
        endif()
    endif()
    if (${MAGNETRON_CPU_APPROX_MATH})
        target_compile_definitions(magnetron PRIVATE MAG_APPROXMATH)
    endif()
    if(${IS_AMD64}) # x86-64 specific compilation options
        target_compile_options(magnetron PRIVATE -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpclmul)
    elseif(${IS_ARM64})
        target_compile_options(magnetron PRIVATE -march=armv8-a+simd)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations only for release builds
        target_compile_options(magnetron PRIVATE -O3 -flto)
        target_link_options(magnetron PRIVATE -flto)
    endif()
endif()

if (${MAGNETRON_ENABLE_CUDA})
    find_package(CUDAToolkit)
    if (CUDAToolkit_FOUND)
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "52;61;70;75")
        endif()
        if (NOT DEFINED CMAKE_CUDA_COMPILER)
            set(CMAKE_CUDA_COMPILER "${MAGNETRON_CUDA_COMPILER}")
        endif()
        set(CMAKE_CUDA_STANDARD 20)
        enable_language(CUDA)
        file(GLOB_RECURSE MAGNETRON_CUDA_SOURCES magnetron/*.cu magnetron/*.cuh)
        add_library(magnetron_cuda SHARED ${MAGNETRON_CUDA_SOURCES})
        target_include_directories(magnetron_cuda PRIVATE extern)
        target_link_libraries(magnetron_cuda CUDA::cudart CUDA::cuda_driver)
        target_compile_definitions(magnetron_cuda PRIVATE MAG_ENABLE_CUDA)
        target_compile_definitions(magnetron PRIVATE MAG_ENABLE_CUDA)
        target_link_libraries(magnetron magnetron_cuda)
    else()
        message(WARNING "CUDA not found, disabling CUDA support")
        set(MAGNETRON_ENABLE_CUDA OFF)
    endif()
endif()

if (${MAGNETRON_BUILD_TESTS})
    add_subdirectory(test)
endif()

if (${MAGNETRON_BUILD_FUZZERS})
    add_subdirectory(fuzzer)
endif()

if (${MAGNETRON_BUILD_BENCHMARKS})
    add_subdirectory(benchmark)
endif()

