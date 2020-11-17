cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
project(VisualOdometry)

option(USE_CUDA "Using CUDA to accelerate" OFF)

if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif(USE_CUDA)

# find_package (Eigen3 REQUIRED NO_MODULE)
# include_directories (${EIGEN3_INCLUDE_DIR})
# message(STATUS "EIGENPATH: " ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
