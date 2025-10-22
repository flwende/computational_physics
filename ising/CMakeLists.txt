include_directories(include ${CMAKE_SOURCE_DIR}/random/include)

add_subdirectory(src)

if (NOT WIN32)
	add_subdirectory(verification)
endif()
