cmake_minimum_required(VERSION 3.10.0)

project(ComputationalPhysics)

set (CMAKE_CXX_STANDARD 17)

add_compile_options(-march=native -Wall -Wno-ignored-attributes)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_BINARY_DIR}/../utility)

add_subdirectory(ising)
add_subdirectory(random)
