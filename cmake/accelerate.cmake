# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

find_library(ACCELERATE_FRAMEWORK Accelerate)

if (ACCELERATE_FRAMEWORK)
    message(STATUS "Accelerate framework found")
    target_compile_definitions(magnetron PRIVATE MAG_ACCELERATE)
    target_compile_definitions(magnetron PRIVATE ACCELERATE_NEW_LAPACK)
    target_compile_definitions(magnetron PRIVATE ACCELERATE_LAPACK_ILP64)
    target_link_libraries(magnetron ${ACCELERATE_FRAMEWORK})
else()
    message(WARNING "Accelerate framework not found, using fallback")
endif()
