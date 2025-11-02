cmake_minimum_required(VERSION 3.10.0)

project(ComputationalPhysics)

set (CMAKE_CXX_STANDARD 20)
#set (CMAKE_BUILD_TYPE Release)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	add_compile_options(-Wall -Wno-ignored-attributes)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	add_compile_options(-Wall -Wno-ignored-attributes)
	if (DEFINED ENV{AMD_GPU_ARCH})
		# For host code compilation: e.g., __GFX11__
		add_compile_definitions($ENV{AMD_GPU_ARCH})
		include_directories ("/usr/include/hip")
	endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	add_compile_options(/W4 /Zi /arch:AVX512)
endif()

if (WIN32)
	add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

if (NOT APPLE)
	add_compile_definitions(USE_SIMD_INTRINSICS)
endif()

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utility)

add_subdirectory(ising)
add_subdirectory(random)
add_subdirectory(utility)
