cmake_minimum_required(VERSION 3.10.0)

project(ComputationalPhysics)

set (CMAKE_CXX_STANDARD 20)

add_compile_options(-march=native -Wall -Wno-ignored-attributes)
if (DEFINED ENV{GPU_ARCH})
  # For host code compilation: e.g., __GFX11__
  add_compile_definitions($ENV{GPU_ARCH})
endif()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_BINARY_DIR}/../utility)

add_subdirectory(ising)
add_subdirectory(random)
add_subdirectory(utility)
