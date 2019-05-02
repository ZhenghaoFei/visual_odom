#ifndef VISUAL_ODOMETER_H
#define VISUAL_ODOMETER_H

// ROS include
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/CameraInfo.h"
#include "nav_msgs/Odometry.h"
#include "std_msgs/Time.h"

#include <cv_bridge/cv_bridge.h>
#include <tf/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "feature.h"
#include "utils.h"
#include "evaluate_odometry.h"
#include "visualOdometry.h"
#include "Frame.h"
#include <math.h>


class VisualOdometer
{
public:

    VisualOdometer(ros::NodeHandle& nh);
    ~VisualOdometer();
    void imageGrabCallback(const sensor_msgs::ImageConstPtr& left_image_msg_ptr, const sensor_msgs::ImageConstPtr& right_image_msg_ptr, 
                           const sensor_msgs::CameraInfoConstPtr& left_cam_info_msg_ptr, const sensor_msgs::CameraInfoConstPtr& right_cam_info_msg_ptr);

    cv::Mat transInCameraFrame(cv::Mat& rotation, cv::Mat& translation);

    void tracking(cv::Mat& image_left, cv::Mat& image_right, ros::Time& stamp);

    bool checkValidTrans(cv::Mat& rotation, cv::Mat& translation);


    void integrateOdometry(cv::Mat& frame_pose, cv::Mat& trans);
    void constructOdomMsg(ros::Time stamp, cv::Mat& frame_pose, cv::Mat& rotation, cv::Mat& translation, float dt);
    void staticTF();

private:
    ros::Publisher pub_odom;
    
    cv::Mat projMatrl;
    cv::Mat projMatrr;
    bool projection_mat_initialized;
    
    cv::Mat image_left_last,  image_right_last;
    ros::Time stamp_last;
    bool vo_initialized;


    FeatureSet currentVOFeatures;
    std::vector<FeaturePoint> oldFeaturePointsLeft;
    std::vector<FeaturePoint> currentFeaturePointsLeft;

    // -----------------------------------------
    // Initialize variables
    // -----------------------------------------
    // cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
    // cv::Mat translation_stereo = cv::Mat::zeros(3, 1, CV_64F);

    cv::Mat pose = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat Rpose = cv::Mat::eye(3, 3, CV_64F);
    
    cv::Mat frame_pose = cv::Mat::eye(4, 4, CV_64F);

    cv::Mat trajectory = cv::Mat::zeros(600, 1200, CV_8UC3);
    cv::Mat points4D, points3D;
    int frame_id = 0;

};


#endif

