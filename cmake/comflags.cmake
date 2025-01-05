# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

if(WIN32) # Windows (MSVC) specific config

    target_compile_options(magnetron PRIVATE /W3 /Oi)
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations for release builds
        target_compile_options(magnetron PRIVATE /O2 /Oy)
    endif()

else() # GCC/Clang specific config

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

    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations only for release builds
        target_compile_options(magnetron PRIVATE -O3 -flto)
        target_link_options(magnetron PRIVATE -flto)
    endif()

    if (${MAGNETRON_BUILD_SHARED})
        target_compile_definitions(magnetron PRIVATE MAG_SHARED)
    endif()
    if (${MAGNETRON_CPU_APPROX_MATH})
        target_compile_definitions(magnetron PRIVATE MAG_APPROXMATH)
    endif()
    if (${MAGNETRON_DEBUG})
        target_compile_definitions(magnetron PRIVATE MAG_DEBUG)
    endif()
    if(${IS_AMD64}) # x86-64 specific compilation options
        target_compile_options(magnetron PRIVATE -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpclmul)
    elseif(${IS_ARM64})
        target_compile_options(magnetron PRIVATE -march=armv8-a+simd)
    endif()
endif()