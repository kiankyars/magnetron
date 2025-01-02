# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE BENCHMARK_SOURCES *.cpp)
add_executable(magnetron_benchmark ${BENCHMARK_SOURCES})
target_link_libraries(magnetron_benchmark magnetron)
target_include_directories(magnetron_benchmark PRIVATE ../magnetron)
target_include_directories(magnetron_benchmark PRIVATE nanobench)
