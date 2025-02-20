# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

option(MI_BUILD_SHARED "" OFF)
option(MI_BUILD_STATIC "" ON)
option(MI_BUILD_TESTS "" OFF)
option(MI_OVERRIDE "" OFF)
option(MI_OSX_INTERPOSE "" OFF)
option(MI_OSX_ZONE "" OFF)
option(MI_WIN_REDIRECT "" OFF)
option(MI_NO_USE_CXX "" ON)
option(MI_NO_OPT_ARCH "" ON)
add_subdirectory(extern/mimalloc)
target_compile_definitions(magnetron PRIVATE MAGNETRON_USE_MIMALLOC)
target_link_libraries(magnetron mimalloc-static)
