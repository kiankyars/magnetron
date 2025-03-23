# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

add_subdirectory(unittests)

add_executable(magnetron_sanity_test sanity_test.c)
target_link_libraries(magnetron_sanity_test magnetron)
target_include_directories(magnetron_sanity_test PRIVATE ../magnetron)

add_executable(magnetron_invariant_proof invariant_proof.cpp)
target_link_libraries(magnetron_invariant_proof magnetron)
target_include_directories(magnetron_invariant_proof PRIVATE ../magnetron)
if (NOT WIN32)
    target_compile_options(magnetron_invariant_proof PRIVATE -Ofast -march=native)
endif()
