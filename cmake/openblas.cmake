# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

set(OPENBLAS_INCLUDE_SEARCH_PATHS
    /usr/include
    /usr/include/openblas
    /usr/include/openblas-base
    /usr/local/include
    /usr/local/include/openblas
    /usr/local/include/openblas-base
    /opt/OpenBLAS/include
    $ENV{OpenBLAS_HOME}
    $ENV{OpenBLAS_HOME}/include
)
find_path(OPENBLAS_INC NAMES cblas.h PATHS ${OPENBLAS_INCLUDE_SEARCH_PATHS})
find_library(OPENBLAS_LIB NAMES openblas libopenblas)
if (OPENBLAS_INC AND OPENBLAS_LIB)
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
    include_directories(${OPENBLAS_INC})
    target_link_libraries(magnetron ${OPENBLAS_LIB})
      target_compile_definitions(magnetron PRIVATE MAG_OPENBLAS)
else()
    message(WARNING "OpenBLAS not found, using fallback")
endif()