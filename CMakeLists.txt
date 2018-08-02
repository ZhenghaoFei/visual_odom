cmake_minimum_required(VERSION 3.10)
# set (EIGENPATH "/Users/holly/Programs/eigen/" )
find_package (Eigen3 REQUIRED NO_MODULE)
INCLUDE_DIRECTORIES ( ${EIGEN3_INCLUDE_DIR} )
message(STATUS "EIGENPATH: " ${EIGEN3_INCLUDE_DIR})

set (CMAKE_CXX_STANDARD 11)

project( VisualOdometry )

add_subdirectory(src)
