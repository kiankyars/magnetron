# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(magnetron_benchmark benchmarks.cpp)
target_link_libraries(magnetron_benchmark magnetron)
target_include_directories(magnetron_benchmark PRIVATE ../magnetron)
target_include_directories(magnetron_benchmark PRIVATE ../extern/nanobench/src/include)

add_executable(magnetron_profile profile.cpp)
target_link_libraries(magnetron_profile magnetron)
target_include_directories(magnetron_profile PRIVATE ../magnetron)
