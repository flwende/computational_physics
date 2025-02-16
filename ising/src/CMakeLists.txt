file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp" "${CMAKE_BINARY_DIR}/../random/src/*.cpp")
list(REMOVE_ITEM sources "${CMAKE_BINARY_DIR}/../random/src/test_rng.cpp")

add_executable(Ising2D ${sources})

include_directories(${CMAKE_BINARY_DIR}/../random/include)

target_compile_definitions(Ising2D PUBLIC RANDOM_SHUFFLE_STATE)
target_compile_options(Ising2D PUBLIC -O3 -fopenmp -ftree-vectorize)
target_link_libraries (Ising2D LINK_PUBLIC "m;gomp")
