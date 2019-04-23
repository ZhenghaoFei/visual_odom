
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
#include "Frame.h"

using namespace std;

int main(int argc, char **argv)
{

    // -----------------------------------------
    // Load images and calibration parameters
    // -----------------------------------------
    bool display_ground_truth = false;
    std::vector<Matrix> pose_matrix_gt;
    if(argc == 4)
    {   display_ground_truth = true;
        cerr << "Display ground truth trajectory" << endl;
        // load ground truth pose
        string filename_pose = string(argv[3]);
        pose_matrix_gt = loadPoses(filename_pose);

    }
    if(argc < 3)
    {
        cerr << "Usage: ./run path_to_sequence path_to_calibration [optional]path_to_ground_truth_pose" << endl;
        return 1;
    }

    // Sequence
    string filepath = string(argv[1]);
    cout << "Filepath: " << filepath << endl;

    // Camera calibration
    string strSettingPath = string(argv[2]);
    cout << "Calibration Filepath: " << strSettingPath << endl;

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float bf = fSettings["Camera.bf"];

    cv::Mat projMatrl = (cv::Mat_<float>(3, 4) << fx, 0., cx, 0., 0., fy, cy, 0., 0,  0., 1., 0.);
    cv::Mat projMatrr = (cv::Mat_<float>(3, 4) << fx, 0., cx, bf, 0., fy, cy, 0., 0,  0., 1., 0.);
    cout << "P_left: " << endl << projMatrl << endl;
    cout << "P_right: " << endl << projMatrr << endl;

    // -----------------------------------------
    // Initialize variables
    // -----------------------------------------
    cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat translation_stereo = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat pose = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);
    
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat frame_pose32 = cv::Mat::eye(4, 4, CV_32F);

    std::cout << "frame_pose " << frame_pose << std::endl;
    cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);
    FeatureSet currentVOFeatures;
    cv::Mat points4D, points3D;
    int init_frame_id = 0;

    // ------------------------
    // Load first images
    // ------------------------
    cv::Mat imageLeft_t0_color,  imageLeft_t0;
    loadImageLeft(imageLeft_t0_color,  imageLeft_t0, init_frame_id, filepath);
    
    cv::Mat imageRight_t0_color, imageRight_t0;  
    loadImageRight(imageRight_t0_color, imageRight_t0, init_frame_id, filepath);

    float fps;

    // -----------------------------------------
    // Run visual odometry
    // -----------------------------------------
    clock_t tic = clock();
    std::vector<FeaturePoint> oldFeaturePointsLeft;
    std::vector<FeaturePoint> currentFeaturePointsLeft;

    for (int frame_id = init_frame_id+1; frame_id < 9000; frame_id++)
    {

        std::cout << std::endl << "frame_id " << frame_id << std::endl;
        // ------------
        // Load images
        // ------------
        cv::Mat imageLeft_t1_color,  imageLeft_t1;
        loadImageLeft(imageLeft_t1_color,  imageLeft_t1, frame_id, filepath);        
        cv::Mat imageRight_t1_color, imageRight_t1;  
        loadImageRight(imageRight_t1_color, imageRight_t1, frame_id, filepath);

        std::vector<cv::Point2f> oldPointsLeft_t0 = currentVOFeatures.points;


        std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;  

        matchingFeatures( imageLeft_t0, imageRight_t0,
                          imageLeft_t1, imageRight_t1, 
                          currentVOFeatures,
                          pointsLeft_t0, 
                          pointsRight_t0, 
                          pointsLeft_t1, 
                          pointsRight_t1);  

        imageLeft_t0 = imageLeft_t1;
        imageRight_t0 = imageRight_t1;

        std::vector<cv::Point2f>& currentPointsLeft_t0 = pointsLeft_t0;
        std::vector<cv::Point2f>& currentPointsLeft_t1 = pointsLeft_t1;

        // std::cout << "oldPointsLeft_t0 size : " << oldPointsLeft_t0.size() << std::endl;
        // std::cout << "currentFramePointsLeft size : " << currentPointsLeft_t0.size() << std::endl;
        
        std::vector<cv::Point2f> newPoints;
        std::vector<bool> valid; // valid new points are ture

        // ---------------------
        // Triangulate 3D Points
        // ---------------------
        cv::Mat points3D_t0, points4D_t0;
        cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0,  pointsRight_t0,  points4D_t0);
        cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);
        // std::cout << "points4D_t0 size : " << points4D_t0.size() << std::endl;

        cv::Mat points3D_t1, points4D_t1;
        // std::cout << "pointsLeft_t1 size : " << pointsLeft_t1.size() << std::endl;
        // std::cout << "pointsRight_t1 size : " << pointsRight_t1.size() << std::endl;

        cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t1,  pointsRight_t1,  points4D_t1);
        cv::convertPointsFromHomogeneous(points4D_t1.t(), points3D_t1);

        // std::cout << "points4D_t1 size : " << points4D_t1.size() << std::endl;

        // ---------------------
        // Tracking transfomation
        // ---------------------
        trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0, pointsLeft_t1, points3D_t0, rotation, translation_stereo);
        displayTracking(imageLeft_t1, pointsLeft_t0, pointsLeft_t1);


        points4D = points4D_t0;
        frame_pose.convertTo(frame_pose32, CV_32F);
        points4D = frame_pose32 * points4D;
        cv::convertPointsFromHomogeneous(points4D.t(), points3D);

        // ------------------------------------------------
        // Intergrating and display
        // ------------------------------------------------

        cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);
        // std::cout << "rotation: " << rotation_euler << std::endl;
        // std::cout << "translation: " << translation_stereo.t() << std::endl;

        cv::Mat rigid_body_transformation;

        if(abs(rotation_euler[1])<0.1 && abs(rotation_euler[0])<0.1 && abs(rotation_euler[2])<0.1)
        {
            integrateOdometryStereo(frame_id, rigid_body_transformation, frame_pose, rotation, translation_stereo);

        } else {

            std::cout << "Too large rotation"  << std::endl;
        }

        // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;

        // std::cout << "frame_pose" << frame_pose << std::endl;


        Rpose =  frame_pose(cv::Range(0, 3), cv::Range(0, 3));
        cv::Vec3f Rpose_euler = rotationMatrixToEulerAngles(Rpose);
        // std::cout << "Rpose_euler" << Rpose_euler << std::endl;

        cv::Mat pose = frame_pose.col(3).clone();

        clock_t toc = clock();
        fps = float(frame_id-init_frame_id)/(toc-tic)*CLOCKS_PER_SEC;

        // std::cout << "Pose" << pose.t() << std::endl;
        std::cout << "FPS: " << fps << std::endl;

        display(frame_id, trajectory, pose, pose_matrix_gt, fps, display_ground_truth);

    }

    return 0;
}

