cmake_minimum_required(VERSION 3.10)

find_package( OpenCV REQUIRED )
find_package(PCL 1.5 REQUIRED)

add_subdirectory(evaluate)

include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(evaluate)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_library(feature SHARED "feature.cpp")
add_library(bucket SHARED "bucket.cpp")
add_library(utils SHARED "utils.cpp")
add_library(visualOdometry SHARED "visualOdometry.cpp")


add_library(MapPoint SHARED "MapPoint.cpp")

add_library(Frame SHARED "Frame.cpp")


set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})
message(${PROJECT_SOURCE_DIR})

add_executable( run main.cpp )
target_link_libraries( bucket ${OpenCV_LIBS} )
target_link_libraries( feature ${OpenCV_LIBS} bucket)
target_link_libraries( utils evaluate_odometry feature ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries( visualOdometry MapPoint utils bucket feature ${OpenCV_LIBS})


target_link_libraries( Frame ${OpenCV_LIBS} )

target_link_libraries( MapPoint Frame ${OpenCV_LIBS} )

target_link_libraries( run ${OpenCV_LIBS} bucket feature utils visualOdometry Frame MapPoint ${PCL_LIBRARIES})
