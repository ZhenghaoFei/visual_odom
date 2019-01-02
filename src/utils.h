#ifndef UTILS_H
#define UTILS_H

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common_headers.h>

#include "feature.h"
#include "matrix.h"
#include "MapPoint.h"


typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// --------------------------------
// Visualization
// --------------------------------
void drawFeaturePoints(cv::Mat image, std::vector<cv::Point2f>& points);

void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose, std::vector<Matrix>& pose_matrix_gt, float fps, bool showgt);

void featureSetToPointClouds(cv::Mat& points3D,  PointCloud::Ptr cloud);

void featureSetToPointCloudsValid(cv::Mat& points3D,  PointCloud::Ptr cloud, std::vector<bool>& valid);

void mapPointsToPointCloudsAppend(std::vector<MapPoint>& mapPoints,  PointCloud::Ptr cloud);

void simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer);


// --------------------------------
// Transformation
// --------------------------------
void integrateOdometryStereo(int frame_id, cv::Mat& rigid_body_transformation, cv::Mat& frame_pose, const cv::Mat& rotation, 
                            const cv::Mat& translation_stereo);

bool isRotationMatrix(cv::Mat &R);

cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R);

// --------------------------------
// I/O
// --------------------------------

void loadImageLeft(cv::Mat& image_color, cv::Mat& image_gary, int frame_id, std::string filepath);

void loadImageRight(cv::Mat& image_color, cv::Mat& image_gary, int frame_id, std::string filepath);

void loadGyro(std::string filename, std::vector<std::vector<double>>& time_gyros);
// read time gyro txt file with format of timestamp, gx, gy, gz

#endif
