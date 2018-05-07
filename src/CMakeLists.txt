cmake_minimum_required(VERSION 3.10)

find_package( OpenCV REQUIRED )
set(EIGENPATH "/Users/holly/Programs/eigen/" )
set (CMAKE_CXX_STANDARD 11)

add_subdirectory(evaluate)

include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(${EIGENPATH} )
include_directories(evaluate)


add_library(feature STATIC "feature.cpp")
add_library(bucket STATIC "bucket.cpp")
add_library(utils STATIC "utils.cpp")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
message(${PROJECT_SOURCE_DIR})

add_executable( run main.cpp )
target_link_libraries( utils evaluate_odometry)
target_link_libraries( run ${OpenCV_LIBS} bucket feature utils)
