file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp" "${CMAKE_SOURCE_DIR}/random/src/*.cpp" "${CMAKE_SOURCE_DIR}/utility/*.cpp")
list(REMOVE_ITEM sources "${CMAKE_SOURCE_DIR}/random/src/test_rng.cpp")
list(REMOVE_ITEM sources "${CMAKE_SOURCE_DIR}/utility/synchronization/test_barrier.cpp")
list(REMOVE_ITEM sources "${CMAKE_SOURCE_DIR}/utility/execution/test_thread_group.cpp")

add_executable(Ising2D ${sources})

include_directories(${CMAKE_SOURCE_DIR}/random/include)

target_compile_definitions(Ising2D PUBLIC RANDOM_SHUFFLE_STATE)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
target_compile_options(Ising2D PUBLIC -O3 -march=native -fopenmp-simd -ftree-vectorize -ffast-math -Rpass=loop-vectorize -save-temps)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#target_compile_options(Ising2D PUBLIC -O3 -march=znver4 -mavx2 -mno-avx512f -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized -save-temps)
target_compile_options(Ising2D PUBLIC -O3 -march=native -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized -save-temps)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
target_compile_options(Ising2D PUBLIC /Ox /arch:AVX512 /openmp:experimental /fp:fast)
endif()

target_link_options(Ising2D PUBLIC)
