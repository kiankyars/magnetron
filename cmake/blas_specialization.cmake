if (${IS_AMD64}) # x86-64 specific compilation options

    set(MAGNETRON_AMD64_SOURCES
        magnetron/magnetron_cpu_blas_amd64_sse41.c
        magnetron/magnetron_cpu_blas_amd64_sse41.c
        magnetron/magnetron_cpu_blas_amd64_avx.c
        magnetron/magnetron_cpu_blas_amd64_avx2.c
        magnetron/magnetron_cpu_blas_amd64_avx512f.c
    )
    set(MAGNETRON_SOURCES ${MAGNETRON_SOURCES} ${MAGNETRON_AMD64_SOURCES})

     set_property(
         SOURCE "${CMAKE_SOURCE_DIR}/magnetron/magnetron_cpu_blas_amd64_sse41.c"
         APPEND
         PROPERTY COMPILE_FLAGS
         "-msse4.1"
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
 elseif(${IS_ARM64})
     # TODO
 endif()