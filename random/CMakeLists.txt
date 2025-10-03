file(GLOB_RECURSE sources CONFIGURE_DEPENDS "src/*.cpp" "${CMAKE_BINARY_DIR}/../utility/hip/*.cpp")

add_executable(TestRNG ${sources})

include_directories(include)

target_compile_definitions(TestRNG PUBLIC RANDOM_SHUFFLE_STATE __USE_SIMD_INTRINSICS__)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
target_compile_options(TestRNG PUBLIC -O3 -march=native -fopenmp-simd -ftree-vectorize -ffast-math -Rpass=loop-vectorize -save-temps)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#target_compile_options(TestRNG PUBLIC -O3 -march=znver4 -mavx2 -mno-avx512f -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized -save-temps)
target_compile_options(TestRNG PUBLIC -O3 -march=native -fopenmp-simd -ftree-vectorize -ffast-math -fopt-info-vec-optimized -save-temps)
endif()

target_link_libraries (TestRNG LINK_PUBLIC "")
