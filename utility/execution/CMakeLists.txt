file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp")

add_executable(TestThreadGroup ${sources})

target_compile_options(TestThreadGroup PUBLIC -O3)
