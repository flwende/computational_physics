file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp" "${CMAKE_BINARY_DIR}/../random/src/*.cpp" "${CMAKE_BINARY_DIR}/../utility/*.cpp")
list(REMOVE_ITEM sources "${CMAKE_BINARY_DIR}/../random/src/test_rng.cpp")
list(REMOVE_ITEM sources "${CMAKE_BINARY_DIR}/../utility/synchronization/test_barrier.cpp")

add_executable(Ising2D ${sources})

include_directories(${CMAKE_BINARY_DIR}/../random/include)

target_include_directories (Ising2D PUBLIC "/usr/include/hip")
target_compile_definitions(Ising2D PUBLIC RANDOM_SHUFFLE_STATE)
target_compile_options(Ising2D PUBLIC -O3 -fopenmp -ftree-vectorize)

target_link_options(Ising2D PUBLIC -fopenmp)
target_link_libraries (Ising2D LINK_PUBLIC "m;gomp")
