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

find_package(Ceres REQUIRED) 

include_directories(${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
list(APPEND 8POINT_LINK_TARGETS ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})

include_directories(${CMAKE_CURRENT_BINARY_DIR} ${8POINT_EIGEN_DIR} "inc")

add_library(8point
    inc/eight/fundamental.h
    inc/eight/essential.h
    inc/eight/normalize.h
    inc/eight/distance.h
    inc/eight/triangulate.h
    inc/eight/project.h
    inc/eight/select.h
    inc/eight/structure.h
    src/fundamental.cpp
    src/essential.cpp
    src/normalize.cpp
    src/distance.cpp
    src/project.cpp
    src/structure.cpp
)

add_executable(8point_tests
    tests/catch.hpp
    tests/utils.h
    tests/test_pose.cpp
    tests/test_pose_outliers.cpp
    tests/test_triangulation.cpp
    tests/test_structure.cpp
)
target_link_libraries(8point_tests 8point ${8POINT_LINK_TARGETS})


add_executable(8point_defocus
    examples/defocus.cpp    
)
target_link_libraries(8point_defocus 8point ${8POINT_LINK_TARGETS})

add_executable(8point_calibrate
    examples/calibrate.cpp    
)
target_link_libraries(8point_calibrate 8point ${8POINT_LINK_TARGETS})