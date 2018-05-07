cmake_minimum_required(VERSION 2.8)
project( FeatureDetection )
find_package( OpenCV REQUIRED )
set(EIGENPATH "/Users/holly/Programs/eigen/" )
set (CMAKE_CXX_STANDARD 11)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${EIGENPATH} )

add_executable( run main.cpp )
add_library(evaluate_odometry STATIC "evaluate_odometry.cpp")
add_library(matrix STATIC "matrix.cpp")
add_library(bucket STATIC "bucket.cpp")

target_link_libraries( evaluate_odometry matrix)
target_link_libraries( run ${OpenCV_LIBS} evaluate_odometry bucket)
