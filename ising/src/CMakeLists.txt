file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp" "${CMAKE_BINARY_DIR}/../random/src/*.cpp" "${CMAKE_BINARY_DIR}/../utility/*.cpp")
list(REMOVE_ITEM sources "${CMAKE_BINARY_DIR}/../random/src/test_rng.cpp")
list(REMOVE_ITEM sources "${CMAKE_BINARY_DIR}/../utility/synchronization/test_barrier.cpp")
list(REMOVE_ITEM sources "${CMAKE_BINARY_DIR}/../utility/execution/test_thread_group.cpp")

add_executable(Ising2D ${sources})

include_directories(${CMAKE_BINARY_DIR}/../random/include)

target_include_directories (Ising2D PUBLIC "/usr/include/hip")
target_compile_definitions(Ising2D PUBLIC RANDOM_SHUFFLE_STATE __USE_SIMD_INTRINSICS__)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
target_compile_options(Ising2D PUBLIC -O3 -march=native -fopenmp-simd -ftree-vectorize -ffast-math -Rpass=loop-vectorize -save-temps)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#target_compile_options(Ising2D PUBLIC -O3 -march=znver4 -mavx2 -mno-avx512f -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized -save-temps)
target_compile_options(Ising2D PUBLIC -O3 -march=native -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized -save-temps)
endif()

target_link_options(Ising2D PUBLIC)
target_link_libraries (Ising2D LINK_PUBLIC "m")
