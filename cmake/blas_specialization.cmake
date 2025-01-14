if (${IS_AMD64}) # x86-64 specific compilation options

    set(MAGNETRON_AMD64_SOURCES
        magnetron/magnetron_cpu_blas_amd64_sse42.c
        magnetron/magnetron_cpu_blas_amd64_avx.c
        magnetron/magnetron_cpu_blas_amd64_avx2.c
        magnetron/magnetron_cpu_blas_amd64_avx512f.c
    )
    set(MAGNETRON_SOURCES ${MAGNETRON_SOURCES} ${MAGNETRON_AMD64_SOURCES})

    if (WIN32)
        set_property(
             SOURCE "${CMAKE_SOURCE_DIR}/magnetron/magnetron_cpu_blas_amd64_sse42.c"
             APPEND
             PROPERTY COMPILE_FLAGS
             "/arch:SSE4.2"
         )
         set_property(
             SOURCE "${CMAKE_SOURCE_DIR}/magnetron/magnetron_cpu_blas_amd64_avx.c"
             APPEND
             PROPERTY COMPILE_FLAGS
             "/arch:AVX"
         )
         set_property(
             SOURCE "${CMAKE_SOURCE_DIR}/magnetron/magnetron_cpu_blas_amd64_avx2.c"
             APPEND
             PROPERTY COMPILE_FLAGS
             "/arch:AVX2"
         )
         set_property(
             SOURCE "${CMAKE_SOURCE_DIR}/magnetron/magnetron_cpu_blas_amd64_avx512f.c"
             APPEND
             PROPERTY COMPILE_FLAGS
             "/arch:AVX512"
         )
    else()
        set_property(
             SOURCE "${CMAKE_SOURCE_DIR}/magnetron/magnetron_cpu_blas_amd64_sse42.c"
             APPEND
             PROPERTY COMPILE_FLAGS
             "-msse4.2"
         )
         set_property(
             SOURCE "${CMAKE_SOURCE_DIR}/magnetron/magnetron_cpu_blas_amd64_avx.c"
             APPEND
             PROPERTY COMPILE_FLAGS
             "-mavx"
         )
         set_property(
             SOURCE "${CMAKE_SOURCE_DIR}/magnetron/magnetron_cpu_blas_amd64_avx2.c"
             APPEND
             PROPERTY COMPILE_FLAGS
             "-mavx2 -mfma"
         )
         set_property(
             SOURCE "${CMAKE_SOURCE_DIR}/magnetron/magnetron_cpu_blas_amd64_avx512f.c"
             APPEND
             PROPERTY COMPILE_FLAGS
             "-mfma -mavx512f"
         )
    endif()
 elseif(${IS_ARM64})
     # TODO
 endif()