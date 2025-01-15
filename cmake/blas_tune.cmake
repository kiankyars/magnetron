# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

function(set_blas_spec_arch filename posix_arch msvc_arch)
    message(STATUS "BLAS CPU permutation ${filename} ${posix_arch} / ${msvc_arch}")
    if (WIN32)
        set_property(SOURCE "${CMAKE_SOURCE_DIR}/magnetron/${filename}" APPEND PROPERTY COMPILE_FLAGS "${msvc_arch}")
    else()
        set_property(SOURCE "${CMAKE_SOURCE_DIR}/magnetron/${filename}" APPEND PROPERTY COMPILE_FLAGS "${posix_arch}")
    endif()
endfunction()

set(MAGNETRON_BLAS_SPEC_AMD64_SOURCES
    magnetron/magnetron_cpu_blas_amd64_sse42.c
    magnetron/magnetron_cpu_blas_amd64_avx.c
    magnetron/magnetron_cpu_blas_amd64_avx2.c
    magnetron/magnetron_cpu_blas_amd64_avx512f.c
)

set(MAGNETRON_BLAS_SPEC_ARM64_SOURCES
    magnetron/magnetron_cpu_blas_arm64_82.c
)

if (${IS_AMD64}) # x86-64 specific compilation options
    set(MAGNETRON_SOURCES ${MAGNETRON_SOURCES} ${MAGNETRON_BLAS_SPEC_AMD64_SOURCES})
    set_blas_spec_arch("magnetron_cpu_blas_amd64_sse42.c" "-mtune=nehalem -msse4.2"  "/arch:SSE4.2")
    set_blas_spec_arch("magnetron_cpu_blas_amd64_avx.c" "-mtune=sandybridge -mavx"  "/arch:AVX")
    set_blas_spec_arch("magnetron_cpu_blas_amd64_avx2.c" "-mtune=skylake -mavx -mavx2 -mfma -mf16c"  "/arch:AVX2")
    set_blas_spec_arch("magnetron_cpu_blas_amd64_avx512f.c" "-mtune=cannonlake -mavx -mavx2 -mfma -mf16c -mavx512f" "/arch:AVX512")
 elseif(${IS_ARM64})
    set(MAGNETRON_SOURCES ${MAGNETRON_SOURCES} ${MAGNETRON_BLAS_SPEC_ARM64_SOURCES})
    set_blas_spec_arch("magnetron_cpu_blas_arm64_82.c" "-march=armv8.2-a+dotprod+fp16" "")
 endif()