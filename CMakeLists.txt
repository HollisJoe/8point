cmake_minimum_required(VERSION 2.8)

project(8point)

if (WIN32)
	add_definitions("-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
endif()

set(8POINT_LINK_TARGETS)

set(8POINT_EIGEN_DIR "../eigen" CACHE PATH "Where is the include directory of Eigen located")
if (WIN32)
  set(OpenCV_STATIC OFF)
	set(OpenCV_SHARED ON)
endif()
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND 8POINT_LINK_TARGETS ${OpenCV_LIBRARIES})

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${8POINT_EIGEN_DIR} "inc")

add_library(8point
    inc/eight/fundamental.h
    inc/eight/normalize.h
    inc/eight/distance.h
    src/fundamental.cpp
    src/normalize.cpp
    src/distance.cpp
)

add_executable(8point_tests
    tests/catch.hpp
    tests/utils.h
    tests/test_8point.cpp

)
target_link_libraries(8point_tests 8point ${8POINT_LINK_TARGETS})
