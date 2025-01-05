# (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

file(GLOB_RECURSE MAGNETRON_SOURCES magnetron/*.h magnetron/*.c)

if (${MAGNETRON_BUILD_SHARED})
    add_library(magnetron SHARED ${MAGNETRON_SOURCES})
else()
    add_library(magnetron STATIC ${MAGNETRON_SOURCES})
endif()

target_include_directories(magnetron PRIVATE extern)
