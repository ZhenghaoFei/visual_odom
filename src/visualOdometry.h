#ifndef VISUAL_ODOM_H
#define VISUAL_ODOM_H

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

#include "feature.h"
#include "bucket.h"
#include "utils.h"
#include "Frame.h"



void matchingFeatures(cv::Mat& imageLeft_t0, cv::Mat& imageRight_t0,
                      cv::Mat& imageLeft_t1, cv::Mat& imageRight_t1, 
                      FeatureSet& currentVOFeatures,
                      std::vector<cv::Point2f>&  pointsLeft_t0, 
                      std::vector<cv::Point2f>&  pointsRight_t0, 
                      std::vector<cv::Point2f>&  pointsLeft_t1, 
                      std::vector<cv::Point2f>&  pointsRight_t1);


void trackingFrame2Frame(cv::Mat& projMatrl, cv::Mat& projMatrr,
                         std::vector<cv::Point2f>&  pointsLeft_t0,
                         std::vector<cv::Point2f>&  pointsLeft_t1, 
                         cv::Mat& points3D_t0,
                         cv::Mat& rotation,
                         cv::Mat& translation,
                         bool mono_rotation=true);

void displayTracking(cv::Mat& imageLeft_t1, 
                     std::vector<cv::Point2f>&  pointsLeft_t0,
                     std::vector<cv::Point2f>&  pointsLeft_t1);

#endif
