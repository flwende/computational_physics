include_directories(include ${CMAKE_SOURCE_DIR}/random/include)

add_subdirectory(src)

if (NOT (WIN32 OR APPLE))
	add_subdirectory(verification)
endif()
