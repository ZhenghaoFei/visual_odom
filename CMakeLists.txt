cmake_minimum_required(VERSION 3.5)
set (CMAKE_CXX_STANDARD 11)
project(VisualOdometry)


# find_package (Eigen3 REQUIRED NO_MODULE)
# include_directories (${EIGEN3_INCLUDE_DIR})
# message(STATUS "EIGENPATH: " ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
