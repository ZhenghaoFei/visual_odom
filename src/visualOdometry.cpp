#include "visualOdometry.h"


cv::Mat euler2rot(cv::Mat& rotationMatrix, const cv::Mat & euler)
{

  double x = euler.at<double>(0);
  double y = euler.at<double>(1);
  double z = euler.at<double>(2);

  // Assuming the angles are in radians.
  double ch = cos(z);
  double sh = sin(z);
  double ca = cos(y);
  double sa = sin(y);
  double cb = cos(x);
  double sb = sin(x);

  double m00, m01, m02, m10, m11, m12, m20, m21, m22;

  m00 = ch * ca;
  m01 = sh*sb - ch*sa*cb;
  m02 = ch*sa*sb + sh*cb;
  m10 = sa;
  m11 = ca*cb;
  m12 = -ca*sb;
  m20 = -sh*ca;
  m21 = sh*sa*cb + ch*sb;
  m22 = -sh*sa*sb + ch*cb;

  rotationMatrix.at<double>(0,0) = m00;
  rotationMatrix.at<double>(0,1) = m01;
  rotationMatrix.at<double>(0,2) = m02;
  rotationMatrix.at<double>(1,0) = m10;
  rotationMatrix.at<double>(1,1) = m11;
  rotationMatrix.at<double>(1,2) = m12;
  rotationMatrix.at<double>(2,0) = m20;
  rotationMatrix.at<double>(2,1) = m21;
  rotationMatrix.at<double>(2,2) = m22;

  return rotationMatrix;
}

void checkValidMatch(std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& points_return, std::vector<bool>& status, int threshold)
{
    int offset;
    for (int i = 0; i < points.size(); i++)
    {
        offset = std::max(std::abs(points[i].x - points_return[i].x), std::abs(points[i].y - points_return[i].y));
        // std::cout << offset << ", ";

        if(offset > 0)
        {
            status.push_back(false);
        }
        else
        {
            status.push_back(true);
        }
    }
}

void removeInvalidPoints(std::vector<cv::Point2f>& points, const std::vector<bool>& status)
{
    int index = 0;
    for (int i = 0; i < status.size(); i++)
    {
        if (status[i] == false)
        {
            points.erase(points.begin() + index);
        }
        else
        {
            index ++;
        }
    }
}


void visualOdometry(int current_frame_id, std::string filepath,
                    cv::Mat& projMatrl, cv::Mat& projMatrr,
                    cv::Mat& rotation, cv::Mat& translation_mono, cv::Mat& translation_stereo, 
                    cv::Mat& image_left_t0,
                    cv::Mat& image_right_t0,
                    FeatureSet& current_features)
{

    // ------------
    // Load images
    // ------------
    cv::Mat image_left_t1_color,  image_left_t1;
    loadImageLeft(image_left_t1_color,  image_left_t1, current_frame_id + 1, filepath);
    
    cv::Mat image_right_t1_color, image_right_t1;  
    loadImageRight(image_right_t1_color, image_right_t1, current_frame_id + 1, filepath);

    // ----------------------------
    // Feature detection using FAST
    // ----------------------------
    std::vector<cv::Point2f>  points_left_t0, points_right_t0, points_left_t1, points_right_t1, points_left_t0_return;   //vectors to store the coordinates of the feature points

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
                     points_left_t0, points_right_t0, points_left_t1, points_right_t1, points_left_t0_return, current_features);

    std::vector<bool> status;
    checkValidMatch(points_left_t0, points_left_t0_return, status, 1);

    removeInvalidPoints(points_left_t0, status);
    removeInvalidPoints(points_left_t0_return, status);
    removeInvalidPoints(points_left_t1, status);
    removeInvalidPoints(points_right_t0, status);

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

    int iterationsCount = 1000;        // number of Ransac iterations.
    float reprojectionError = 1.0;    // maximum allowed distance to consider it an inlier.
    float confidence = 0.99;          // RANSAC successful confidence.
    bool useExtrinsicGuess = true;
    int flags =cv::SOLVEPNP_ITERATIVE;

    cv::solvePnPRansac( points3D_t0, points_left_t1, intrinsic_matrix, distCoeffs, rvec, translation_stereo,
                        useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                        inliers, flags );

    std::cout << "inliers size: " << inliers.size() << std::endl;

    // translation_stereo = -translation_stereo;

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

    int radius = 2;
    // cv::Mat vis = image_left_t0.clone();

    cv::Mat vis;

    cv::cvtColor(image_left_t1, vis, CV_GRAY2BGR, 3);


    for (int i = 0; i < points_left_t0.size(); i++)
    {
        circle(vis, cvPoint(points_left_t0[i].x, points_left_t0[i].y), radius, CV_RGB(0,255,0));
    }

    for (int i = 0; i < points_left_t1.size(); i++)
    {
        circle(vis, cvPoint(points_left_t1[i].x, points_left_t1[i].y), radius, CV_RGB(255,0,0));
    }

    for (int i = 0; i < points_left_t1.size(); i++)
    {
        cv::line(vis, points_left_t0[i], points_left_t1[i], CV_RGB(0,255,0));
    }

    imshow("vis ", vis );
    
}


