# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

add_subdirectory(unit)

add_executable(magnetron_santity_test santiy_test.c)
target_link_libraries(magnetron_santity_test magnetron)
target_include_directories(magnetron_santity_test PRIVATE ../magnetron)
