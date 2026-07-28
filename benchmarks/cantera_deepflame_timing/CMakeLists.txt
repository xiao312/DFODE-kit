cmake_minimum_required(VERSION 3.16)

project(cantera_deepflame_timing LANGUAGES CXX)

set(CANTERA_ROOT "$ENV{CONDA_PREFIX}" CACHE PATH "Cantera installation prefix")
if(NOT CANTERA_ROOT)
    message(FATAL_ERROR "Set CANTERA_ROOT or activate the Cantera conda environment")
endif()

find_path(
    CANTERA_INCLUDE_DIR
    cantera/base/Solution.h
    HINTS "${CANTERA_ROOT}/include"
    REQUIRED
)
find_library(
    CANTERA_LIBRARY
    NAMES cantera
    HINTS "${CANTERA_ROOT}/lib"
    REQUIRED
)
find_library(
    FMT_LIBRARY
    NAMES fmt
    HINTS "${CANTERA_ROOT}/lib"
)
find_path(
    FMT_INCLUDE_DIR
    fmt/format.h
    HINTS "${CANTERA_ROOT}/include"
)
find_path(
    EIGEN_INCLUDE_DIR
    eigen3/Eigen/Sparse
    HINTS "${CANTERA_ROOT}/include"
)

file(
    STRINGS
    "${CANTERA_INCLUDE_DIR}/cantera/base/config.h"
    CANTERA_VERSION_LINE
    REGEX "^#define CANTERA_VERSION \"[0-9]+\\.[0-9]+\\.[0-9]+\""
)
string(
    REGEX MATCH
    "([0-9]+)\\.([0-9]+)\\.([0-9]+)"
    CANTERA_VERSION_MATCH
    "${CANTERA_VERSION_LINE}"
)
set(CANTERA_VERSION_MAJOR "${CMAKE_MATCH_1}")
set(CANTERA_VERSION_MINOR "${CMAKE_MATCH_2}")
set(CANTERA_VERSION_PATCH "${CMAKE_MATCH_3}")
math(
    EXPR CANTERA_VERSION_NUMBER
    "${CANTERA_VERSION_MAJOR} * 10000 + ${CANTERA_VERSION_MINOR} * 100 + ${CANTERA_VERSION_PATCH}"
)

add_executable(
    cantera_deepflame_timing
    deepflame_cantera_timing.cpp
)

target_compile_features(cantera_deepflame_timing PRIVATE cxx_std_17)
target_compile_options(
    cantera_deepflame_timing
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)
target_include_directories(
    cantera_deepflame_timing
    PRIVATE
        "${CANTERA_INCLUDE_DIR}"
        "${CANTERA_ROOT}/include"
        "${CANTERA_ROOT}/include/eigen3"
        "${FMT_INCLUDE_DIR}"
        "${EIGEN_INCLUDE_DIR}"
)
target_link_libraries(cantera_deepflame_timing PRIVATE "${CANTERA_LIBRARY}")

if(FMT_LIBRARY)
    target_link_libraries(cantera_deepflame_timing PRIVATE "${FMT_LIBRARY}")
endif()

if(CANTERA_VERSION_NUMBER GREATER_EQUAL 30200)
    target_compile_definitions(
        cantera_deepflame_timing
        PRIVATE DFODE_CANTERA_HAS_SOLVER_STATS=1
    )
endif()

message(
    STATUS
    "Cantera ${CANTERA_VERSION_MAJOR}.${CANTERA_VERSION_MINOR}.${CANTERA_VERSION_PATCH}; version number ${CANTERA_VERSION_NUMBER}"
)

set_target_properties(
    cantera_deepflame_timing
    PROPERTIES
        BUILD_RPATH "${CANTERA_ROOT}/lib"
)
