# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

file(GLOB_RECURSE TEST_SOURCES *.hpp *.cpp)
add_executable(magnetron_test ${TEST_SOURCES})
target_link_libraries(magnetron_test magnetron)
target_include_directories(magnetron_test PRIVATE ../../magnetron)
target_include_directories(magnetron_test PRIVATE ../extern/half/include)
target_include_directories(magnetron_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(
    magnetron_test
    GTest::gtest_main
    GTest::gmock
)
add_test(NAME magnetron_test COMMAND magnetron_test)

if (${MAGNETRON_ENABLE_CUDA})
    target_compile_definitions(magnetron_test PRIVATE MAG_ENABLE_CUDA)
endif()
