cmake_minimum_required(VERSION 3.20)
project(numeric LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

# Set default build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Debug
      CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -march=native -Wall -Rpass-analysis=loop-vectorize -fsave-optimization-record"
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Applying Debug flags for macOS leaks")
  set(CMAKE_CXX_FLAGS_DEBUG
      "-g -O0 -fno-omit-frame-pointer \
      -Wall -Wextra"
      CACHE STRING "" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set ARMPL paths (adjust as needed)
set(ARMPL_ROOT "/opt/arm/armpl_24.10_flang-new_clang_19")
set(ARMPL_INCLUDE_DIR "${ARMPL_ROOT}/include")
set(ARMPL_LIB_DIR "${ARMPL_ROOT}/lib")

# Locate ARMPL libraries
find_library(
  ARMPL_BLAS_LIB
  NAMES armpl
  PATHS ${ARMPL_LIB_DIR})
find_library(
  ARMPL_LAPACK_LIB
  NAMES armpl
  PATHS ${ARMPL_LIB_DIR})

# Ensure ARMPL is found
if(ARMPL_BLAS_LIB AND ARMPL_LAPACK_LIB)
  message(STATUS "Found ARMPL BLAS: ${ARMPL_BLAS_LIB}")
  message(STATUS "Found ARMPL LAPACK: ${ARMPL_LAPACK_LIB}")
else()
  message(FATAL_ERROR "ARMPL libraries not found! Check installation.")
endif()

add_library(swnumeric INTERFACE)
enable_testing()

add_subdirectory(Libraries)
add_subdirectory(Applications)

target_link_libraries(swnumeric INTERFACE ${ARMPL_BLAS_LIB} ${ARMPL_LAPACK_LIB})
