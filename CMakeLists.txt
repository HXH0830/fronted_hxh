cmake_minimum_required(VERSION 2.8)
project(fast_optical_flow)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

find_package(OpenCV 3.2)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV not found.")
    endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 3.1.0 REQUIRED)
find_package(glog 0.4.0 REQUIRED)
find_package(gflags 2.2.2 REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${OpenCV_INCLUDE_DIRS}
${GLOG_INCLUDE_DIRS}
${GFLAG_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/fast_optical_flow_front_end.cc
include/fast_optical_flow_front_end.h
)

target_link_libraries(
${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
glog
gflags
)

add_executable(mono_euroc main.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})