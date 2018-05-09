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

#include "feature.h"
#include "matrix.h"


void drawFeaturePoints(cv::Mat image, std::vector<cv::Point2f>& points);

void initializeImagesFeatures(int current_frame_id, std::string filepath, cv::Mat& image_left_t0, cv::Mat& image_right_t0, FeatureSet& features);

cv::Mat loadImageLeft(int frame_id, std::string filepath);

cv::Mat loadImageRight(int frame_id, std::string filepath);

double getAbsoluteScale(int frame_id);

void display(int frame_id, cv::Mat& trajectory, cv::Mat& pose, std::vector<Matrix>& pose_matrix_gt, float fps, bool showgt);

void integrateOdometryMono(int frame_id, cv::Mat& pose, cv::Mat& Rpose, const cv::Mat& rotation, 
                           const cv::Mat& translation_mono);

void integrateOdometryScale(int frame_id, cv::Mat& pose, cv::Mat& Rpose, const cv::Mat& rotation, 
                            const cv::Mat& translation_mono, const cv::Mat& translation_stereo);

void integrateOdometryStereo(int frame_id, cv::Mat& pose, cv::Mat& Rpose, const cv::Mat& rotation, 
                            const cv::Mat& translation_stereo);

void loadGyro(std::string filename, std::vector<std::vector<double>>& time_gyros);
// read time gyro txt file with format of timestamp, gx, gy, gz

#endif
