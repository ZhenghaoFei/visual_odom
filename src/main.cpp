
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
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <opencv2/core/eigen.hpp>

#include "feature.h"
#include "bucket.h"
#include "solver.h"
#include "utils.h"
#include "evaluate_odometry.h"

void visualOdometry(int current_frame_id, std::string filepath,
                    cv::Mat& projMatrl, cv::Mat& projMatrr,
                    cv::Mat& rotation, cv::Mat& translation_mono, cv::Mat& translation_stereo, 
                    cv::Mat& image_left_t0,
                    cv::Mat& image_right_t0,
                    FeatureSet& current_features){

    // ------------
    // Load images
    // ------------
    cv::Mat image_left_t1 = loadImageLeft(current_frame_id + 1, filepath);
    cv::Mat image_right_t1 = loadImageRight(current_frame_id + 1, filepath);

    // ----------------------------
    // Feature detection using FAST
    // ----------------------------
    std::vector<cv::Point2f>  points_left_t0, points_right_t0, points_left_t1, points_right_t1;   //vectors to store the coordinates of the feature points

    if (current_features.size() < 2000)
    {
        // use all new features
        // featureDetectionFast(image_left_t0, current_features.points);     
        // current_features.ages = std::vector<int>(current_features.points.size(), 0);

        // append new features with old features
        appendNewFeatures(image_left_t0, current_features);   

        std::cout << "Current feature set size: " << current_features.points.size() << std::endl;
    }


    // --------------------------------------------------------
    // Feature tracking using KLT tracker, bucketing and circular matching
    // --------------------------------------------------------
    int bucket_size = 50;
    int features_per_bucket = 4;
    bucketingFeatures(image_left_t0, current_features, bucket_size, features_per_bucket);

    points_left_t0 = current_features.points;

    circularMatching(image_left_t0, image_right_t0, image_left_t1, image_right_t1,
                     points_left_t0, points_right_t0, points_left_t1, points_right_t1,
                     current_features);


    current_features.points = points_left_t1;

    // -----------------------------------------------------------
    // Rotation(R) estimation using Nister's Five Points Algorithm
    // -----------------------------------------------------------
    double focal = projMatrl.at<float>(0, 0);
    cv::Point2d principle_point(projMatrl.at<float>(0, 2), projMatrl.at<float>(1, 2));

    //recovering the pose and the essential cv::matrix
    cv::Mat E, mask;
    E = cv::findEssentialMat(points_left_t1, points_left_t0, focal, principle_point, cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, points_left_t1, points_left_t0, rotation, translation_mono, focal, principle_point, mask);

    // ---------------------
    // Triangulate 3D Points
    // ---------------------
    cv::Mat points4D_t0;
    triangulatePoints( projMatrl,  projMatrr,  points_left_t0,  points_right_t0,  points4D_t0);

    // ------------------------------------------------
    // Translation (t) estimation by use solvePnPRansac
    // ------------------------------------------------
    cv::Mat points3D_t0;
    convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64FC1);  
    cv::Mat inliers;  
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat intrinsic_matrix = (cv::Mat_<float>(3, 3) << projMatrl.at<float>(0, 0), projMatrl.at<float>(0, 1), projMatrl.at<float>(0, 2),
                                                 projMatrl.at<float>(1, 0), projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2),
                                                 projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2), projMatrl.at<float>(1, 3));

    int iterationsCount = 500;        // number of Ransac iterations.
    float reprojectionError = 2.0;    // maximum allowed distance to consider it an inlier.
    float confidence = 0.95;          // RANSAC successful confidence.
    bool useExtrinsicGuess = false;
    int flags =cv::SOLVEPNP_ITERATIVE;

    cv::solvePnPRansac( points3D_t0, points_left_t1, intrinsic_matrix, distCoeffs, rvec, translation_stereo,
                        useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                        inliers, flags );

    translation_stereo = -translation_stereo;

    // std::cout << "rvec : " <<rvec <<std::endl;
    // std::cout << "translation_stereo : " <<translation_stereo <<std::endl;

    // -----------------------------------------
    // Prepare image for next frame
    // -----------------------------------------
    image_left_t0 = image_left_t1;
    image_right_t0 = image_right_t1;


    // -----------------------------------------
    // Display
    // -----------------------------------------

    // imshow( "Left camera", image_left_t0 );
    // imshow( "Right camera", image_right_t0 );

    drawFeaturePoints(image_left_t1, current_features.points);
    imshow("Features ", image_left_t1 );
}


int main(int argc, char const *argv[])
{

    // -----------------------------------------
    // Load images and calibration parameters
    // -----------------------------------------
    char filename_pose[200];
    sprintf(filename_pose, "/Users/holly/Downloads/KITTI/poses/00.txt");
    std::string filepath = "/Users/holly/Downloads/KITTI/sequences/00/";
    std::vector<Matrix> pose_matrix_gt = loadPoses(filename_pose);
    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << 718.8560, 0., 607.1928, 0., 0., 718.8560, 185.2157, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << 718.8560, 0., 607.1928, -386.1448, 0., 718.8560, 185.2157, 0., 0,  0., 1., 0.);

    // -----------------------------------------
    // Initialize variables
    // -----------------------------------------
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation_mono = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat translation_stereo = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat pose = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);

    FeatureSet current_features;

    int init_frame_id = 0;
    cv::Mat image_l, image_r;
    float fps;

    // -----------------------------------------
    // Run visual odometry
    // -----------------------------------------
    initializeImagesFeatures(init_frame_id, filepath, image_l, image_r, current_features);

    clock_t tic = clock();

    for (int frame_id = init_frame_id; frame_id < 1000; frame_id++)
    {

        std::cout << std::endl << "frame_id " << frame_id << std::endl;

        visualOdometry(frame_id, filepath,
                       projMatrl, projMatrr,
                       rotation, translation_mono, translation_stereo, 
                       image_l, image_r,
                       current_features);

        // integrateOdometryMono(frame_id, pose, Rpose, rotation, translation_mono);

        // integrateOdometryScale(frame_id, pose, Rpose, rotation, translation_mono, translation_stereo);

        integrateOdometryStereo(frame_id, pose, Rpose, rotation, translation_stereo);

        clock_t toc = clock();
        fps = float(frame_id-init_frame_id)/(toc-tic)*CLOCKS_PER_SEC;

        std::cout << "Pose" << pose.t() << std::endl;
        std::cout << "FPS: " << fps << std::endl;

        display(frame_id, trajectory, pose, pose_matrix_gt, fps);


    }

    return 0;
}

