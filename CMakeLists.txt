# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

cmake_minimum_required(VERSION 3.18)

project(magnetron LANGUAGES C)

message("Configuring magnetron project for ${CMAKE_SYSTEM_PROCESSOR}...")

set(CMAKE_C_STANDARD 99) # Use C99 standard
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

option(MAGNETRON_BUILD_TESTS "Build tests" ON)                                  # Build tests
option(MAGNETRON_BUILD_BENCHMARKS "Build benchmarks" ON)                        # Build benchmarks
option(MAGNETRON_BUILD_FUZZERS "Build fuzzers" OFF)                             # (Experimental) Build fuzzers

option(MAGNETRON_DEBUG "Enable debug mode" OFF)                                 # Enable debug assertions, bound checks and other debug features. (Always enabled in Debug builds)
option(MAGNETRON_CPU_APPROX_MATH "Trade precision for performance" ON)          # (CPU only) Enable SIMD math function approximations. Greatly increases performance. Try disabling if you encounter numerical instability. Does NOT enable -ffast-math or similar compiler flags.


option(MAGNETRON_ENABLE_CUDA "Enable CUDA support" OFF)                          # Enable CUDA support
option(MAGNETRON_ENABLE_MIMALLOC "Use mimalloc as memory allocator" ON)         # Use mimalloc as memory allocator for faster memory allocation

set(MAGNETRON_CUDA_COMPILER "/usr/local/cuda-12.6/bin/nvcc" CACHE STRING "Path to the CUDA compiler") # Set to your CUDA compiler path
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

if (${MAGNETRON_BUILD_TESTS} OR ${MAGNETRON_BUILD_BENCHMARKS} OR ${MAGNETRON_BUILD_FUZZERS})
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(cmake/arch.cmake)
include(cmake/lib.cmake)
include(cmake/compiler_config.cmake)

if (${MAGNETRON_ENABLE_CUDA})
    include(cmake/cuda.cmake)
endif()

if (${MAGNETRON_ENABLE_MIMALLOC})
    include(cmake/allocator.cmake)
endif()

if (${MAGNETRON_BUILD_TESTS})
    enable_testing()
    add_subdirectory(test)
endif()

if (${MAGNETRON_BUILD_FUZZERS})
    add_subdirectory(fuzzer)
endif()

if (${MAGNETRON_BUILD_BENCHMARKS})
    add_subdirectory(benchmark)
endif()

message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Configuring magnetron project for ${CMAKE_SYSTEM_PROCESSOR}... done")

install(TARGETS magnetron LIBRARY DESTINATION . RUNTIME DESTINATION .)
