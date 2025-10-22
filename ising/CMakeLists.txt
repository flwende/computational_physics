include_directories(include ${CMAKE_SOURCE_DIR}/random/include)

add_subdirectory(src)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	add_subdirectory(verification)
endif()
