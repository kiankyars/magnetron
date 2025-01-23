# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

function(set_blas_spec_arch filename posix_arch msvc_arch)
    message(STATUS "BLAS CPU permutation ${filename} ${posix_arch} / ${msvc_arch}")
    if (WIN32)
        set_property(SOURCE "${CMAKE_SOURCE_DIR}/magnetron/${filename}" APPEND PROPERTY COMPILE_FLAGS "${msvc_arch}")
    else()
        set_property(SOURCE "${CMAKE_SOURCE_DIR}/magnetron/${filename}" APPEND PROPERTY COMPILE_FLAGS "${posix_arch}")
    endif()
endfunction()

set(MAGNETRON_BLAS_SPEC_AMD64_SOURCES
    magnetron/magnetron_cpu_blas_amd64_v2.c
    magnetron/magnetron_cpu_blas_amd64_v2_5.c
    magnetron/magnetron_cpu_blas_amd64_v3.c
    magnetron/magnetron_cpu_blas_amd64_v4.c
    magnetron/magnetron_cpu_blas_amd64_v4_5.c
)

set(MAGNETRON_BLAS_SPEC_ARM64_SOURCES
    magnetron/magnetron_cpu_blas_arm64_v8_2.c
    magnetron/magnetron_cpu_blas_arm64_v9.c
)

if (${IS_AMD64}) # x86-64 specific compilation options
    set(MAGNETRON_SOURCES ${MAGNETRON_SOURCES} ${MAGNETRON_BLAS_SPEC_AMD64_SOURCES})
    set_blas_spec_arch("magnetron_cpu_blas_amd64_v2.c" "-mtune=nehalem -mcx16 -mpopcnt -msse3 -mssse3 -msse4.1 -msse4.2"  "/arch:SSE4.2")
    set_blas_spec_arch("magnetron_cpu_blas_amd64_v2_5.c" "-mtune=ivybridge -mavx -mno-avx2 -mcx16 -mpopcnt -msse3 -mssse3 -msse4.1 -msse4.2"  "/arch:AVX")
    set_blas_spec_arch("magnetron_cpu_blas_amd64_v3.c" "-mtune=haswell -mavx -mavx2 -mbmi -mbmi2 -mf16c -mfma -mlzcnt -mmovbe"  "/arch:AVX2")
    set_blas_spec_arch("magnetron_cpu_blas_amd64_v4.c" "-mtune=cannonlake -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx -mavx2 -mbmi -mbmi2 -mf16c -mfma -mlzcnt -mmovbe"  "/arch:AVX512")
    set_blas_spec_arch("magnetron_cpu_blas_amd64_v4_5.c" "-mtune=generic -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -mavx512bf16 -mavx -mavx2 -mbmi -mbmi2 -mf16c -mfma -mlzcnt -mmovbe" "/arch:AVX512")
 elseif(${IS_ARM64})
    set(MAGNETRON_SOURCES ${MAGNETRON_SOURCES} ${MAGNETRON_BLAS_SPEC_ARM64_SOURCES})
    set_blas_spec_arch("magnetron_cpu_blas_arm64_v8_2.c" "-march=armv8.2-a+dotprod+fp16" "")
    set_blas_spec_arch("magnetron_cpu_blas_arm64_v9.c" "-march=armv9-a+sve+sve2" "")
 endif()