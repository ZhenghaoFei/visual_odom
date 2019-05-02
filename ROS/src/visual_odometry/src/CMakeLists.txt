set(visual_odom_src "../../../../src/")
message(STATUS "visual_odom_src: " ${visual_odom_src})

find_package( OpenCV REQUIRED )
find_package (Eigen3 REQUIRED NO_MODULE)


include_directories(${visual_odom_src}) 
include_directories(${visual_odom_src}/evaluate) 
include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories (${EIGEN3_INCLUDE_DIR})


add_library(evaluate_odometry SHARED ${visual_odom_src}evaluate/evaluate_odometry.cpp)
add_library(matrix SHARED ${visual_odom_src}evaluate/matrix.cpp)
add_library(feature SHARED ${visual_odom_src}feature.cpp)
add_library(bucket SHARED ${visual_odom_src}bucket.cpp)
add_library(utils SHARED ${visual_odom_src}utils.cpp)
add_library(visualOdometry SHARED ${visual_odom_src}visualOdometry.cpp)
add_library(Frame SHARED ${visual_odom_src}Frame.cpp)


target_link_libraries( evaluate_odometry matrix)
target_link_libraries( bucket ${OpenCV_LIBS} )
target_link_libraries( feature ${OpenCV_LIBS} bucket)
target_link_libraries( utils evaluate_odometry feature ${OpenCV_LIBS} )
target_link_libraries( visualOdometry utils bucket feature ${OpenCV_LIBS})
target_link_libraries( Frame ${OpenCV_LIBS} )


add_library(VisualOdometer SHARED "VisualOdometer.cpp")
target_link_libraries (VisualOdometer ${catkin_LIBRARIES} ${OpenCV_LIBS} bucket feature utils visualOdometry Frame)

add_executable(runVisualOdometry "visual_odom_node.cpp")
target_link_libraries(runVisualOdometry  VisualOdometer ${catkin_LIBRARIES})

# add_dependencies(runPointcloudPreProcessor template_in_row_localization_generate_messages_cpp)
