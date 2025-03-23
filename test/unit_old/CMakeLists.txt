# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

file(GLOB_RECURSE TEST_SOURCES *.cpp)
add_executable(magnetron_test ${TEST_SOURCES})
target_link_libraries(magnetron_test magnetron)
target_include_directories(magnetron_test PRIVATE ../../magnetron)
target_link_libraries(
    magnetron_test
    GTest::gtest_main
)
add_test(NAME magnetron_test COMMAND magnetron_test)

if (${MAGNETRON_ENABLE_CUDA})
    target_compile_definitions(magnetron_test PRIVATE MAG_ENABLE_CUDA)
endif()
