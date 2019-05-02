#include "VisualOdometer.h"


VisualOdometer::VisualOdometer(ros::NodeHandle& nh)
{
    // publisher
    pub_odom = nh.advertise<nav_msgs::Odometry> ("/vo/odom", 1);
    projection_mat_initialized = false;
    vo_initialized = false;
    staticTF();
}

void VisualOdometer::staticTF()
{
    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
    geometry_msgs::TransformStamped static_transformStamped;

    static_transformStamped.header.stamp = ros::Time::now();
    static_transformStamped.header.frame_id = "odom";
    static_transformStamped.child_frame_id = "camera_odom";
    static_transformStamped.transform.translation.x = 0.;
    static_transformStamped.transform.translation.y = 0.;
    static_transformStamped.transform.translation.z = 0.;
    tf2::Quaternion quat;
    quat.setRPY(-M_PI/2, 0, -M_PI/2);
    static_transformStamped.transform.rotation.x = quat.x();
    static_transformStamped.transform.rotation.y = quat.y();
    static_transformStamped.transform.rotation.z = quat.z();
    static_transformStamped.transform.rotation.w = quat.w();
    static_broadcaster.sendTransform(static_transformStamped);
}

cv::Mat VisualOdometer::transInCameraFrame(cv::Mat& rotation, cv::Mat& translation)
{
    // The visual odometey solvePnp give rotation and translation in camera frame (z forwarding),
    // transform it into x forawrd

    cv::Mat_<double> T_in_cam_opt = (cv::Mat_<double>(4, 4) << rotation.at<double>(0), rotation.at<double>(1), rotation.at<double>(2), translation.at<double>(0),
                                                               rotation.at<double>(3), rotation.at<double>(4), rotation.at<double>(5), translation.at<double>(1),
                                                               rotation.at<double>(6), rotation.at<double>(7), rotation.at<double>(8), translation.at<double>(2),
                                                               0,  0,  0, 1);

    cv::Mat_<double> cam_T_cam_opt = (cv::Mat_<double>(4, 4) <<   0,  0, 1, 0,
                                                                   -1, 0, 0, 0,
                                                                   0, -1, 0, 0,
                                                                   0,  0, 0, 1);

    cv::Mat_<double> T_in_cam = cam_T_cam_opt * T_in_cam_opt * cam_T_cam_opt.inv();

    return T_in_cam;

}


void VisualOdometer::integrateOdometry(cv::Mat& frame_pose, cv::Mat& trans)
{
    frame_pose = frame_pose * trans;
}

void VisualOdometer::constructOdomMsg(ros::Time stamp, cv::Mat& frame_pose, cv::Mat& rotation, cv::Mat& translation, float dt)
{

    cv::Mat pose_rotation_matrix = frame_pose(cv::Range(0, 3), cv::Range(0, 3));
    // std::cout << "pose_rotation_matrix " << pose_rotation_matrix << std::endl;

    cv::Vec3f pose_rotation_euler = rotationMatrixToEulerAngles(pose_rotation_matrix);
    std::cout << "pose_rotation_euler " << pose_rotation_euler << std::endl;

    geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromRollPitchYaw(double(pose_rotation_euler[0]), double(pose_rotation_euler[1]), double(pose_rotation_euler[2]));
    geometry_msgs::TransformStamped odom_trans;

    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "odom";

    //set the position
    odom.pose.pose.position.x = frame_pose.at<double>(3);
    odom.pose.pose.position.y = frame_pose.at<double>(7);
    odom.pose.pose.position.z = frame_pose.at<double>(11);
    odom.pose.pose.orientation = odom_quat;

    //set the velocity
    odom.child_frame_id = "camera";
    // odom.twist.twist.linear.x = vx;
    // odom.twist.twist.linear.y = vy;
    // odom.twist.twist.angular.z = vth;
    pub_odom.publish(odom);

}


bool VisualOdometer::checkValidTrans(cv::Mat& rotation, cv::Mat& translation)
{

    bool valid = true;

    // Check rotation
    cv::Vec3f rotation_euler = rotationMatrixToEulerAngles(rotation);

    if(abs(rotation_euler[1])>0.1 || abs(rotation_euler[0])>0.1 || abs(rotation_euler[2])>0.1)
    {
        std::cout << "[WARNING] Too large rotation" << std::endl;
        valid = false;
    }

    // Check translation
    double scale = sqrt((translation.at<double>(3))*(translation.at<double>(3)) 
                        + (translation.at<double>(7))*(translation.at<double>(7))
                        + (translation.at<double>(11))*(translation.at<double>(11))) ;

    std::cout << "scale: " << scale << std::endl;

    if (scale <= 0.0 || scale > 10) 
    {
        std::cout << "[WARNING] scale below 0.1, or incorrect translation" << std::endl;
        valid = false;
    }

    return valid;
}


void VisualOdometer::tracking(cv::Mat& image_left, cv::Mat& image_right, ros::Time& stamp)
{   frame_id += 1;
    if(!vo_initialized)
    {
        image_left_last = image_left;
        image_right_last = image_right;
        stamp_last = stamp;
        vo_initialized = true;
        std::cout << "vo initialized " << std::endl;
    }
    else
    {
        std::vector<cv::Point2f> pointsLeft_t0, pointsRight_t0, pointsLeft_t1, pointsRight_t1;  
        std::vector<cv::Point2f> oldPointsLeft_t0 = currentVOFeatures.points;

        // std::cout << " image_left_last height width: " << image_left_last.rows << " " <<image_left_last.cols << std::endl;

    

        matchingFeatures( image_left_last, image_right_last,
                          image_left, image_right, 
                          currentVOFeatures,
                          pointsLeft_t0, 
                          pointsRight_t0, 
                          pointsLeft_t1, 
                          pointsRight_t1,
                          50,
                          2);  

        float delta_t = (stamp - stamp_last).toSec();
        image_left_last = image_left;
        image_right_last = image_right;
        stamp_last = stamp;
        std::cout << "delta_t : " << delta_t << std::endl;
        std::cout << "currentFramePointsLeft size : " << pointsLeft_t0.size() << std::endl;

        if (pointsLeft_t0.size() > 100)
        {

            // ---------------------
            // Triangulate 3D Points
            // ---------------------
            cv::Mat points3D_t0, points4D_t0;
            cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t0,  pointsRight_t0,  points4D_t0);
            cv::convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);

            cv::Mat points3D_t1, points4D_t1;
            cv::triangulatePoints( projMatrl,  projMatrr,  pointsLeft_t1,  pointsRight_t1,  points4D_t1);
            cv::convertPointsFromHomogeneous(points4D_t1.t(), points3D_t1);

            // ---------------------
            // Tracking transfomation
            // ---------------------
            cv::Mat rotation = cv::Mat::eye(3, 3, CV_64F);
            cv::Mat translation = cv::Mat::zeros(3, 1, CV_64F);


            // rotation = (cv::Mat_<double>(3, 3) <<   0, 0, 1,
            //                                        -0, 1, 0,
            //                                        -1, 0, 0);

            // translation = (cv::Mat_<double>(3, 1) << 1, 2, 3);
            // std::cout << "rotation: " << rotation << std::endl;
            // std::cout << "translation: " << translation << std::endl;


            // std::cout << "trans: " << trans << std::endl;


            trackingFrame2Frame(projMatrl, projMatrr, pointsLeft_t0, pointsLeft_t1, points3D_t0, rotation, translation);
            cv::Mat trans = transInCameraFrame(rotation, translation);

            // translation = -translation;

            // displayTracking(image_left, pointsLeft_t0, pointsLeft_t1);
            // cv::waitKey(1);


            // ------------------------------------------------
            // Intergrating and display
            // ------------------------------------------------
            bool trans_valid = checkValidTrans(rotation, translation);


            // std::cout << "rotation: " << rotation_euler << std::endl;
            // std::cout << "translation: " << translation.t() << std::endl;

            if(trans_valid)
            {
                 integrateOdometry(frame_pose, trans);
                 constructOdomMsg(stamp, frame_pose, rotation, translation, delta_t);

            }

            std::cout << "frame_pose" << frame_pose << std::endl;


            // // std::cout << "rigid_body_transformation" << rigid_body_transformation << std::endl;

            // // std::cout << "frame_pose" << frame_pose << std::endl;


            // Rpose =  frame_pose(cv::Range(0, 3), cv::Range(0, 3));
            // cv::Vec3f Rpose_euler = rotationMatrixToEulerAngles(Rpose);
            // // std::cout << "Rpose_euler" << Rpose_euler << std::endl;

            // cv::Mat pose = frame_pose.col(3).clone();


            // std::cout << "Pose: " << pose.t() << std::endl;

        }
        else
        {
            std::cerr << "[]WARNING] Matched feature cirtically low" << pose.t() << std::endl;

        }

    }


}

void VisualOdometer::imageGrabCallback(const sensor_msgs::ImageConstPtr& left_image_msg_ptr, const sensor_msgs::ImageConstPtr& right_image_msg_ptr,
                                       const sensor_msgs::CameraInfoConstPtr& left_cam_info_msg_ptr, const sensor_msgs::CameraInfoConstPtr& right_cam_info_msg_ptr)

{

    std::cout << std::endl;
    ros::Time stamp = left_image_msg_ptr->header.stamp;
    std::cout << "Recived Image Pair at: " << stamp << std::endl;
    // std::cout << "left: " << left_image_msg_ptr->header.stamp << std::endl;
    // std::cout << "right: " << right_image_msg_ptr->header.stamp << std::endl;
    
    if(!projection_mat_initialized)
    {
        projMatrl = cv::Mat(3, 4, CV_64FC1, (void *) left_cam_info_msg_ptr->P.data());
        projMatrr = cv::Mat(3, 4, CV_64FC1, (void *) right_cam_info_msg_ptr->P.data());
        projMatrl.convertTo(projMatrl, CV_32F);
        projMatrr.convertTo(projMatrr, CV_32F);

        std::cout << "projMatrl " << projMatrl << std::endl;
        std::cout << "projMatrr " << projMatrr << std::endl;  
        projection_mat_initialized = true;      
    }
    else
    {
        // Copy the ros image message to cv::Mat.
        cv_bridge::CvImageConstPtr cv_ptrLeft;
        try
        {
            cv_ptrLeft = cv_bridge::toCvShare(left_image_msg_ptr);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv_bridge::CvImageConstPtr cv_ptrRight;
        try
        {
            cv_ptrRight = cv_bridge::toCvShare(right_image_msg_ptr);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }


        cv::Mat image_left,  image_right;
        cvtColor(cv_ptrLeft->image, image_left, cv::COLOR_BGR2GRAY);
        cvtColor(cv_ptrRight->image, image_right, cv::COLOR_BGR2GRAY);

        // mpSLAM->TrackStereo(cv_ptrLeft->image,cv_ptrRight->image,cv_ptrLeft->header.stamp.toSec());
        tracking(image_left, image_right, stamp);
    }
}

VisualOdometer::~VisualOdometer()
{

}

