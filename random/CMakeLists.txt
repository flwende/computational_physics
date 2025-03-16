file(GLOB_RECURSE sources CONFIGURE_DEPENDS "src/*.cpp" "${CMAKE_BINARY_DIR}/../utility/hip/*.cpp")

add_executable(TestRNG ${sources})

include_directories(include)

target_compile_definitions(TestRNG PUBLIC RANDOM_SHUFFLE_STATE)
target_compile_options(TestRNG PUBLIC -O3 -fopenmp-simd -ftree-vectorize -ffast-math)
target_link_libraries (TestRNG LINK_PUBLIC "")
