file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.cpp" "${CMAKE_SOURCE_DIR}/utility/misc/*.cpp")

add_executable(TestThreadGroup ${sources})

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
target_compile_options(Ising2D PUBLIC
	$<$<CONFIG:Debug>:/openmp:experimental /fp:precise>
    $<$<CONFIG:Release>:/Ox /openmp:experimental /fp:fast>
)
endif()
