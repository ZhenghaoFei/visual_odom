
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
#include "utils.h"
#include "evaluate_odometry.h"
#include "visualOdometry.h"

int main(int argc, char const *argv[])
{



    // -----------------------------------------
    // Load images and calibration parameters
    // -----------------------------------------
    char filename_pose[200];
    sprintf(filename_pose, "/Users/holly/Downloads/KITTI/poses/00.txt");
    std::vector<Matrix> pose_matrix_gt = loadPoses(filename_pose);
    std::string filepath = "/Users/holly/Downloads/KITTI/sequences/00/";
    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << 718.8560, 0., 607.1928, 0., 0., 718.8560, 185.2157, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << 718.8560, 0., 607.1928, -386.1448, 0., 718.8560, 185.2157, 0., 0,  0., 1., 0.);

    // sprintf(filename_pose, "/Users/holly/Downloads/KITTIRAW/sequence/poses/00.txt");
    // std::vector<Matrix> pose_matrix_gt = loadPoses(filename_pose);
    // std::string filepath = "/Users/holly/Downloads/KITTIRAW/sequence/";
    // cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << 721.6377, 0., 609.5593, 0., 0., 721.6377, 172.8540, 0., 0,  0., 1., 0.);
    // cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << 721.6377, 0., 609.5593, -387.5744, 0., 721.6377, 172.8540, 0., 0,  0., 1., 0.);


    // -----------------------------------------
    // Load IMU's gyro data and timestamp
    // -----------------------------------------    std::string filename
    std::string gryopath = "/Users/holly/Downloads/KITTIRAW/sequence/oxts/time_gyro.txt";
    std::vector<std::vector<double>> time_gyros;
    loadGyro(gryopath, time_gyros);

    // -----------------------------------------
    // Initialize variables
    // -----------------------------------------
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation_mono = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat translation_stereo = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat pose = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);
    
    // double init_yaw = 1.887805 - M_PI/2;
    // cv::Mat Rpose = (cv::Mat_<double>(3, 3) << cos(init_yaw), 0, sin(init_yaw), 0, 1,  0, -sin(init_yaw), 0, cos(init_yaw));
    // std::cout << "Init Rpose" << Rpose << std::endl;

    cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);

    FeatureSet current_features;

    int init_frame_id = 0;
    cv::Mat image_l, image_r;
    float fps;
    bool show_gt = true;

    // -----------------------------------------
    // Run visual odometry
    // -----------------------------------------
    initializeImagesFeatures(init_frame_id, filepath, image_l, image_r, current_features);

    clock_t tic = clock();

    for (int frame_id = init_frame_id; frame_id < 9000; frame_id++)
    {

        std::cout << std::endl << "frame_id " << frame_id << std::endl;

        visualOdometry(frame_id, filepath,
                       projMatrl, projMatrr,
                       rotation, translation_mono, translation_stereo, 
                       image_l, image_r,
                       current_features);

        // visualOdometryIMU(frame_id, filepath,
        //                projMatrl, projMatrr,
        //                rotation, translation_mono, translation_stereo, 
        //                image_l, image_r,
        //                current_features,
        //                time_gyros);

        // integrateOdometryMono(frame_id, pose, Rpose, rotation, translation_mono);

        // integrateOdometryScale(frame_id, pose, Rpose, rotation, translation_mono, translation_stereo);

        integrateOdometryStereo(frame_id, pose, Rpose, rotation, translation_stereo);

        clock_t toc = clock();
        fps = float(frame_id-init_frame_id)/(toc-tic)*CLOCKS_PER_SEC;

        std::cout << "Pose" << pose.t() << std::endl;
        std::cout << "FPS: " << fps << std::endl;

        display(frame_id, trajectory, pose, pose_matrix_gt, fps, show_gt);


    }

    return 0;
}

