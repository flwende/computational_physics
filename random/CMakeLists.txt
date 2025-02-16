file(GLOB_RECURSE sources CONFIGURE_DEPENDS "src/*.cpp")

add_executable(TestRNG ${sources})

include_directories(include)

target_compile_options(TestRNG PUBLIC -O3 -fopenmp -ftree-vectorize)
target_link_libraries (TestRNG LINK_PUBLIC "gomp")
