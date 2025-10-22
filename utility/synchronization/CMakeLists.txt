file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")

add_executable(TestBarrier ${sources})

target_compile_options(TestBarrier PUBLIC -O3)
