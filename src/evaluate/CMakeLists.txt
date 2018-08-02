cmake_minimum_required(VERSION 3.10)

add_library(evaluate_odometry SHARED "evaluate_odometry.cpp")
add_library(matrix SHARED "matrix.cpp")
target_link_libraries( evaluate_odometry matrix)
