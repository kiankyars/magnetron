# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#[===[

    -moutline-atomics
        Outline-atomics is a gcc compilation flag that adds runtime detection on if the cpu supports atomic instructions.
        Some older ARM CPU's such as the chip on the Raspberry PI 4 don't support atomic instructions. Using them will result in a SIGILL.
        When the outline-atomics flag is used, the compiler will generate code that checks if the CPU supports atomic instructions at runtime.
        CPUs that don't support atomic instructions will use the old load-exclusive/store-exclusive instructions.
        If a different compilation flag defined an architecture that unconditionally supports atomic instructions (e.g. -march=armv8.2), the outline-atomic flag will have no effect.
]===]

message("Configuring magnetron project for ${CMAKE_SYSTEM_PROCESSOR}...")
message("C compiler: ${CMAKE_C_COMPILER_ID}")

set(MAG_MSVC_COMPILE_FLAGS
    /W3
    /Oi
    /arch:SSE2
)
set(MAG_MSVC_RELEASE_COMPILE_FLAGS
    /O2
    /Oy
    /Ot
    /Ob3
)
set(MAG_MSVC_LINK_OPTIONS "")
set(MAG_MSVC_RELEASE_LINK_OPTIONS "")

set(MAG_CLANG_COMPILE_FLAGS
    -std=c99
    -std=gnu99
    -fvisibility=hidden
    -fno-math-errno
    -Wall
    -Werror
    -Wno-error=overflow
    -Wno-error=unused-function
)
set(MAG_CLANG_RELEASE_COMPILE_FLAGS
    -O3
    -flto
    -fomit-frame-pointer
)
set(MAG_CLANG_LINK_OPTIONS "")
set(MAG_CLANG_RELEASE_LINK_OPTIONS -flto)

set(MAG_GCC_COMPILE_FLAGS
    -std=c99
    -std=gnu99
    -fvisibility=hidden
    -fno-math-errno
    -Wall
    -Werror
    -Wno-error=overflow
    -Wno-error=unused-function
    -Wno-error=format-truncation
)
set(MAG_GCC_RELEASE_COMPILE_FLAGS
    -O3
    -flto=auto
    -fomit-frame-pointer
)
set(MAG_GCC_LINK_OPTIONS "")
set(MAG_GCC_RELEASE_LINK_OPTIONS -flto=auto)

if (${IS_ARM64})
    if (NOT WIN32)
        add_compile_options(-moutline-atomics)
    endif()
    set(MAG_CLANG_COMPILE_FLAGS ${MAG_CLANG_COMPILE_FLAGS} -march=armv8-a -moutline-atomics) # See beginning for file for info of -moutline-atomics
    set(MAG_GCC_COMPILE_FLAGS ${MAG_CLANG_COMPILE_FLAGS} -march=armv8-a -moutline-atomics)
elseif (${IS_AMD64})
    set(MAG_CLANG_COMPILE_FLAGS ${MAG_CLANG_COMPILE_FLAGS} -msse -msse2)
    set(MAG_GCC_COMPILE_FLAGS ${MAG_CLANG_COMPILE_FLAGS} -msse -msse2)
endif()

if (WIN32) # Windows (MSVC) specific config
    target_compile_options(magnetron PRIVATE ${MAG_MSVC_COMPILE_FLAGS})
    target_link_options(magnetron PRIVATE ${MAG_MSVC_LINK_OPTIONS})
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations for release builds
        target_compile_options(magnetron PRIVATE ${MAG_MSVC_RELEASE_COMPILE_FLAGS})
        target_link_options(magnetron PRIVATE ${MAG_MSVC_RELEASE_LINK_OPTIONS})
    endif()
else() # GCC/Clang specific config
    target_link_libraries(magnetron m) # link math library
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(magnetron PRIVATE ${MAG_GCC_COMPILE_FLAGS})
        target_link_options(magnetron PRIVATE ${MAG_GCC_LINK_OPTIONS})
    else()
        target_compile_options(magnetron PRIVATE ${MAG_CLANG_COMPILE_FLAGS})
        target_link_options(magnetron PRIVATE ${MAG_CLANG_LINK_OPTIONS})
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")     # Enable optimizations only for release builds
        if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
            target_compile_options(magnetron PRIVATE ${MAG_GCC_RELEASE_COMPILE_FLAGS})
            target_link_options(magnetron PRIVATE ${MAG_GCC_RELEASE_LINK_OPTIONS})
        else()
            target_compile_options(magnetron PRIVATE ${MAG_CLANG_RELEASE_COMPILE_FLAGS})
            target_link_options(magnetron PRIVATE ${MAG_CLANG_RELEASE_LINK_OPTIONS})
        endif()
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
endif()