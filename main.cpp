
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <opencv2/core/eigen.hpp>

#include "evaluate_odometry.h"
#include "feature_set.h"

using Eigen::MatrixXd;
using namespace cv;

// Generic function
template<typename _Scalar, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>

struct functor
{
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

    int m_inputs, m_values;

    functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

};

void hom2cart2( Eigen::Matrix<double, 3,  Eigen::Dynamic> & points3_l,  Eigen::Matrix<double, 3,  Eigen::Dynamic> & points3_r){
  // convert homogeneous coordinates to cartesian coordinates by normalizing z
  for (int i = 0; i < points3_l.cols(); i++)
  {
      points3_l(0, i) = points3_l(0, i)/points3_l(2, i);
      points3_l(1, i) = points3_l(1, i)/points3_l(2, i);
      points3_l(2, i) = 1.;

      points3_r(0, i) = points3_r(0, i)/points3_r(2, i);
      points3_r(1, i) = points3_r(1, i)/points3_r(2, i);
      points3_r(2, i) = 1.;
  }
}

void hom2cart( Eigen::Matrix<double, 3,  Eigen::Dynamic> & points3){
  // convert homogeneous coordinates to cartesian coordinates by normalizing z
  for (int i = 0; i < points3.cols(); i++)
  {
      points3(0, i) = points3(0, i)/points3(2, i);
      points3(1, i) = points3(1, i)/points3(2, i);
      points3(2, i) = 1.;
  }
}


struct reprojection_error_function : functor<double>
// Reprojection error function to be minimized for translation t
// K1, K2: 3 x 3 Intrinsic parameters matrix for the left and right cameras
// R: 3x3 Estimated rotation matrix from the previous step
// points3D: 3xM 3D Point cloud generated from stereo pair in left camera
// frame
// pts_l: matched feature points locations in left camera frame
// pts_r: matched feature points locations in right camera frame
{

    Eigen::Matrix<double, 3, 4> P1;
    Eigen::Matrix<double, 3, 4> P2;
    Eigen::Matrix<double, 3, 3> R;
    Eigen::Matrix4d T;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> points4D_l;
    std::vector<Point2f> points_l_t1;
    std::vector<Point2f> points_r_t1;
    int function_nums;

    reprojection_error_function(Eigen::Matrix<double, 3, 4> proj_0, 
                                Eigen::Matrix<double, 3, 4> proj_1, 
                                Eigen::Matrix<double, 3, 3> rotation,
                                Eigen::Matrix4d T_left2right,
                                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> points4D_t0_eigen,
                                std::vector<Point2f> points_left_t1, 
                                std::vector<Point2f> points_right_t1,
                                int size
                                ): functor<double>(3, size*2) 
    {
         P1 = proj_0;
         P2 = proj_1;
         R = rotation;
         T = T_left2right;
         points4D_l = points4D_t0_eigen;
         points_l_t1 = points_left_t1;
         points_r_t1 = points_right_t1;
         function_nums = size;
    }


    int operator()(const Eigen::VectorXd &translation, Eigen::VectorXd &fvec) const
    {
        // Implement y = 10*(x0+3)^2 + (x1-5)^2
        // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> points4D_r;
        Eigen::Matrix<double, 4, 4> rigid_transformation;

        Eigen::Matrix<double, 3,  Eigen::Dynamic> projection_left;
        Eigen::Matrix<double, 3,  Eigen::Dynamic> projection_right;

        // points4D_r = T * points4D_l;

        rigid_transformation << R(0, 0), R(0, 1), R(0, 2), translation(0),
                                R(1, 0), R(1, 1), R(1, 2), translation(1),
                                R(2, 0), R(2, 1), R(2, 2), translation(2),
                                     0.,      0.,      0.,             1.;

        // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> aa = rigid_transformation * points4D_l;




        projection_left = P1 * rigid_transformation * points4D_l;
        // projection_right = P2 * rigid_transformation * points4D_l; 

        // hom2cart2(projection_left, projection_right);
        hom2cart(projection_left);

        for (int i = 0; i < function_nums; i++)
        {
            int feature_idx = 2 * i;
            fvec(feature_idx) =  projection_left(0, i) - double(points_l_t1[i].x); 
            fvec(feature_idx + 1) =  projection_left(1, i) - double(points_l_t1[i].y);
            // fvec(feature_idx + 2) =  projection_right(0, i) - double(points_r_t1[i].x); 
            // fvec(feature_idx + 3) =  projection_right(1, i) - double(points_r_t1[i].y); 


            // fvec(feature_idx) =  translation(1) + 5;
            // fvec(feature_idx + 1) =  translation(2) + 5;
            // fvec(feature_idx + 2) =  translation(0) + 5;
            // fvec(feature_idx + 3) =  translation(1) + 5;
        }

        return 0;
    }
};


double getAbsoluteScale(int frame_id)    {
  std::string line;
  int i = 0;
  std::ifstream myfile ("/Users/holly/Downloads/KITTI/poses/00.txt");
  double x =0, y=0, z = 0;
  double x_prev, y_prev, z_prev;
  if (myfile.is_open())
  {
    while (( std::getline (myfile,line) ) && (i<=frame_id))
    {
      z_prev = z;
      x_prev = x;
      y_prev = y;
      std::istringstream in(line);
      //cout << line << '\n';
      for (int j=0; j<12; j++)  {
        in >> z ;
        if (j==7) y=z;
        if (j==3)  x=z;
      }
      
      i++;
    }
    myfile.close();
  }

  else {
    std::cout << "Unable to open file";
    return 0;
  }

  return sqrt((x-x_prev)*(x-x_prev) + (y-y_prev)*(y-y_prev) + (z-z_prev)*(z-z_prev)) ;

}

void deleteUnmatchFeatures(std::vector<Point2f>& points0, std::vector<Point2f>& points1, std::vector<uchar>& status){
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  Point2f pt = points1.at(i- indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))   
        {
              if((pt.x<0)||(pt.y<0))    
              {
                status.at(i) = 0;
              }
              points0.erase (points0.begin() + (i - indexCorrection));
              points1.erase (points1.begin() + (i - indexCorrection));
              indexCorrection++;
        }

     }
}

void featureDetection(Mat image, std::vector<Point2f>& points)  {   //uses FAST as of now, modify parameters as necessary
  std::vector<KeyPoint> keypoints;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  cv::FAST(image, keypoints, fast_threshold, nonmaxSuppression);
  KeyPoint::convert(keypoints, points, std::vector<int>());
}

void featureTracking(Mat img_1, Mat img_2, std::vector<Point2f>& points1, std::vector<Point2f>& points2, std::vector<uchar>& status)   { 
//this function automatically gets rid of points for which tracking fails

  std::vector<float> err;                    
  Size winSize=Size(21,21);                                                                                             
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
  deleteUnmatchFeatures(points1, points2, status);

}


void deleteUnmatchFeaturesCircle(std::vector<Point2f>& points0, std::vector<Point2f>& points1,
                          std::vector<Point2f>& points2, std::vector<Point2f>& points3,
                          std::vector<uchar>& status0, std::vector<uchar>& status1,
                          std::vector<uchar>& status2, std::vector<uchar>& status3,
                          std::vector<int>& ages){
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  for (int i = 0; i < ages.size(); ++i)
  {
     ages[i] += 1;
  }

  int indexCorrection = 0;
  for( int i=0; i<status3.size(); i++)
     {  Point2f pt0 = points0.at(i- indexCorrection);
        Point2f pt1 = points1.at(i- indexCorrection);
        Point2f pt2 = points2.at(i- indexCorrection);
        Point2f pt3 = points3.at(i- indexCorrection);
        
        if ((status3.at(i) == 0)||(pt3.x<0)||(pt3.y<0)||
            (status2.at(i) == 0)||(pt2.x<0)||(pt2.y<0)||
            (status1.at(i) == 0)||(pt1.x<0)||(pt1.y<0)||
            (status0.at(i) == 0)||(pt0.x<0)||(pt0.y<0))   
        {
          if((pt0.x<0)||(pt0.y<0)||(pt1.x<0)||(pt1.y<0)||(pt2.x<0)||(pt2.y<0)||(pt3.x<0)||(pt3.y<0))    
          {
            status3.at(i) = 0;
          }
          points0.erase (points0.begin() + (i - indexCorrection));
          points1.erase (points1.begin() + (i - indexCorrection));
          points2.erase (points2.begin() + (i - indexCorrection));
          points3.erase (points3.begin() + (i - indexCorrection));
          ages.erase (ages.begin() + (i - indexCorrection));
          indexCorrection++;
        }

     }  
}

void circularMatching(Mat img_l_0, Mat img_r_0, Mat img_l_1, Mat img_r_1,
                      std::vector<Point2f>& points_l_0, std::vector<Point2f>& points_r_0,
                      std::vector<Point2f>& points_l_1, std::vector<Point2f>& points_r_1,
                      FeatureSet& current_features) { 
  
  //this function automatically gets rid of points for which tracking fails

  std::vector<float> err;                    
  Size winSize=Size(21,21);                                                                                             
  TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);

  std::vector<uchar> status0;
  std::vector<uchar> status1;
  std::vector<uchar> status2;
  std::vector<uchar> status3;


  calcOpticalFlowPyrLK(img_l_0, img_r_0, points_l_0, points_r_0, status0, err, winSize, 3, termcrit, 0, 0.001);
 
  calcOpticalFlowPyrLK(img_r_0, img_r_1, points_r_0, points_r_1, status1, err, winSize, 3, termcrit, 0, 0.001);
    
  calcOpticalFlowPyrLK(img_r_1, img_l_1, points_r_1, points_l_1, status2, err, winSize, 3, termcrit, 0, 0.001);
    
  calcOpticalFlowPyrLK(img_l_1, img_l_0, points_l_1, points_l_0, status3, err, winSize, 3, termcrit, 0, 0.001);
  
  std::cout << "points : " << points_l_0.size() << " "<< points_r_0.size() << " "<< points_r_1.size() << " "<< points_l_1.size() << " "<<std::endl;

  deleteUnmatchFeaturesCircle(points_l_0, points_r_0, points_r_1, points_l_1,
                        status0, status1, status2, status3, current_features.ages);

  std::cout << "points : " << points_l_0.size() << " "<< points_r_0.size() << " "<< points_r_1.size() << " "<< points_l_1.size() << " "<<std::endl;

}


void drawFeaturePoints(Mat image, std::vector<Point2f>& points){
    int radius = 2;
    
    for (int i = 0; i < points.size(); i++)
    {
        circle(image, cvPoint(points[i].x, points[i].y), radius, CV_RGB(255,255,255));
    }
}


Mat loadImageLeft(int frame_id){
    char filename[200];
    sprintf(filename, "/Users/holly/Downloads/KITTI/sequences/00/image_0/%06d.png", frame_id);
    Mat image = imread(filename);
    cvtColor(image, image, COLOR_BGR2GRAY);

    return image;
}

Mat loadImageRight(int frame_id){
    char filename[200];
    sprintf(filename, "/Users/holly/Downloads/KITTI/sequences/00/image_1/%06d.png", frame_id);
    Mat image = imread(filename);
    cvtColor(image, image, COLOR_BGR2GRAY);

    return image;
}

void bucketingFeatures(Mat& image, std::vector<Point2f>& current_features, int bucket_size, int features_per_bucket){
// This function buckets features
// image: only use for getting dimension of the image
// bucket_size: bucket size in pixel is bucket_size*bucket_size
// features_per_bucket: number of selected features per bucket
    int image_height = image.rows;
    int image_width = image.cols;

    int buckets_nums_height = image_height/bucket_size;
    int buckets_nums_width = image_width/bucket_size;


    for (int buckets_idx_height = 0; buckets_idx_height < buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width < buckets_nums_width; buckets_idx_width++)
      {
        /* code */
      }
    }


}

void initializeImagesFeatures(int current_frame_id, 
                        Mat& image_left_t0, Mat& image_right_t0,
                        FeatureSet& features){

    image_left_t0 = loadImageLeft(current_frame_id);
    image_right_t0 = loadImageRight(current_frame_id);

    featureDetection(image_left_t0, features.points);        

    for(int i = 0; i < features.points.size(); i++)
    {
      features.ages.push_back(0);
    }

}

void appendNewFeatures(Mat& image, FeatureSet& current_features){
    
    std::vector<Point2f>  points_new;
    featureDetection(image, points_new);


    current_features.points.insert(current_features.points.end(), points_new.begin(), points_new.end());

    std::vector<int>  ages_new(points_new.size(), 0);
    current_features.ages.insert(current_features.ages.end(), ages_new.begin(), ages_new.end());


}

void visualOdometry(int current_frame_id,
                    Mat& projMatrl, Mat& projMatrr,
                    Mat& rotation, Mat& translation_mono, Mat& translation_stereo, 
                    Mat& image_left_t0,
                    Mat& image_right_t0,
                    FeatureSet& current_features
                    // std::vector<Point2f>& current_features, 
                    // std::vector<Point2f>& points_left_save,
                    // std::vector<Point2f>& points_right_save,
                    // std::vector<int>& feature_ages
                    ){

    // ------------
    // Load images
    // ------------
    Mat image_left_t1 = loadImageLeft(current_frame_id + 1);
    Mat image_right_t1 = loadImageRight(current_frame_id + 1);

    // ----------------------------
    // Feature detection using FAST
    // ----------------------------
    std::vector<Point2f>  points_left_t0, points_right_t0, points_left_t1, points_right_t1;   //vectors to store the coordinates of the feature points

    if (current_features.points.size() <= 2000)
    {
        std::cout << "Reinitialize feature set: "  << std::endl;

        // use all new features
        featureDetection(image_left_t0, current_features.points);     
        current_features.ages = std::vector<int>(current_features.points.size(), 0);

        // append new features with old features
        // appendNewFeatures(image_left_t0, current_features);   
    }   

    std::cout << "current feature set size: " << current_features.points.size() << std::endl;

    // --------------------------------------------------------
    // Feature tracking using KLT tracker and circular matching
    // --------------------------------------------------------

    points_left_t0 = current_features.points;

    circularMatching(image_left_t0, image_right_t0, image_left_t1, image_right_t1,
                     points_left_t0, points_right_t0, points_left_t1, points_right_t1,
                     current_features);

    std::cout << "current_features.ages size: " << current_features.ages.size() << std::endl;
    current_features.points = points_left_t1;

    // std::cout << "current_features.ages " << std::endl;
    // for(int i = 0; i < current_features.ages.size(); i++)
    // {
    //   std::cout   << current_features.ages[i] << ", "; 
    // }
    // std::cout << std::endl;


    // -----------------------------------------------------------
    // Rotation(R) estimation using Nister's Five Points Algorithm
    // -----------------------------------------------------------
    double focal = projMatrl.at<float>(0, 0);
    cv::Point2d principle_point(projMatrl.at<float>(0, 2), projMatrl.at<float>(1, 2));
    // std::cout << "focal " << focal << std::endl;
    // std::cout << "principle_point: " << principle_point << std::endl;

    //recovering the pose and the essential matrix
    Mat E, mask;
    E = findEssentialMat(points_left_t1, points_left_t0, focal, principle_point, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points_left_t1, points_left_t0, rotation, translation_mono, focal, principle_point, mask);


    // ---------------------
    // Triangulate 3D Points
    // ---------------------
    Mat points4D_t0;

    triangulatePoints( projMatrl,  projMatrr,  points_left_t0,  points_right_t0,  points4D_t0);

    // ------------------------------------------------
    // Translation (t) estimation by use solvePnPRansac
    // ------------------------------------------------
    Mat points3D_t0;
    convertPointsFromHomogeneous(points4D_t0.t(), points3D_t0);
    Mat distCoeffs = Mat::zeros(4, 1, CV_64FC1);  
    Mat inliers;  
    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
    Mat intrinsic_matrix = (Mat_<float>(3, 3) << projMatrl.at<float>(0, 0), projMatrl.at<float>(0, 1), projMatrl.at<float>(0, 2),
                                                 projMatrl.at<float>(1, 0), projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2),
                                                 projMatrl.at<float>(1, 1), projMatrl.at<float>(1, 2), projMatrl.at<float>(1, 3));

    int iterationsCount = 500;        // number of Ransac iterations.
    float reprojectionError = 2.0;    // maximum allowed distance to consider it an inlier.
    float confidence = 0.95;          // RANSAC successful confidence.
    bool useExtrinsicGuess = false;
    int flags =SOLVEPNP_ITERATIVE;

    cv::solvePnPRansac( points3D_t0, points_left_t1, intrinsic_matrix, distCoeffs, rvec, translation_stereo,
                        useExtrinsicGuess, iterationsCount, reprojectionError, confidence,
                        inliers, flags );

    translation_stereo = -translation_stereo;

    // std::cout << "rvec : " <<rvec <<std::endl;
    // std::cout << "translation_stereo : " <<translation_stereo <<std::endl;


    // -----------------------------------------
    // Prepare image for next frame
    // -----------------------------------------
    // points_left_save = points_left_t1;
    // points_right_save = points_right_t1;
    image_left_t0 = image_left_t1;
    image_right_t0 = image_right_t1;



    // imshow( "Left camera", image_left_t0 );
    // imshow( "Right camera", image_right_t0 );


    drawFeaturePoints(image_left_t1, current_features.points);
    imshow("points ", image_left_t1 );



}

void integrateOdometryMono(int frame_id, Mat& pose, Mat& Rpose, const Mat& rotation, const Mat& translation_mono)
{
    double scale = 1.00;
    scale = getAbsoluteScale(frame_id);

    std::cout << "translation_mono: " << scale*translation_mono.t() << std::endl;

    if ((scale>0.1)&&(translation_mono.at<double>(2) > translation_mono.at<double>(0)) && (translation_mono.at<double>(2) > translation_mono.at<double>(1))) {
    // if ((scale>0.1)) {

        pose = pose + scale * Rpose * translation_mono;
        Rpose = rotation * Rpose;
    }
    
    else {
     std::cout << "[WARNING] scale below 0.1, or incorrect translation" << std::endl;
    }


}

void integrateOdometryScale(int frame_id, Mat& pose, Mat& Rpose, const Mat& rotation, const Mat& translation_mono, const Mat& translation_stereo)
{

    double scale = sqrt((translation_stereo.at<double>(0))*(translation_stereo.at<double>(0)) 
                        + (translation_stereo.at<double>(1))*(translation_stereo.at<double>(1))
                        + (translation_stereo.at<double>(2))*(translation_stereo.at<double>(2))) ;

    // if (scale<10) {
    if ((scale>0.1)&&(translation_mono.at<double>(2) > translation_mono.at<double>(0)) && (translation_mono.at<double>(2) > translation_mono.at<double>(1))) {


      pose = pose +scale *  Rpose * translation_mono;
      Rpose = rotation * Rpose;

    }

}

void integrateOdometryStereo(int frame_id, Mat& pose, Mat& Rpose, const Mat& rotation, const Mat& translation_stereo)
{
    if ((translation_stereo.at<double>(2)<10)) {


      pose = pose + Rpose * translation_stereo;
      Rpose = rotation * Rpose;

    }

}

void display(int frame_id, Mat& trajectory, Mat& pose, std::vector<Matrix>& pose_matrix_gt)
{
    Mat pose_gt = Mat::zeros(1, 3, CV_64F);
    
    pose_gt.at<double>(0) = pose_matrix_gt[frame_id].val[0][3];
    pose_gt.at<double>(1) = pose_matrix_gt[frame_id].val[0][7];
    pose_gt.at<double>(2) = pose_matrix_gt[frame_id].val[0][11];

    int x = int(pose.at<double>(0)) + 300;
    int y = int(pose.at<double>(2)) + 100;
    circle(trajectory, Point(x, y) ,1, CV_RGB(255,0,0), 2);

    x = int(pose_gt.at<double>(0)) + 300;
    y = int(pose_gt.at<double>(2)) + 100;
    circle(trajectory, Point(x, y) ,1, CV_RGB(255,255,0), 2);

    imshow( "Trajectory", trajectory );


    waitKey(1);
}

int main(int argc, char const *argv[])
{

    char filename_pose[200];
    sprintf(filename_pose, "/Users/holly/Downloads/KITTI/poses/00.txt");
    std::vector<Matrix> pose_matrix_gt = loadPoses(filename_pose);
    Mat projMatrl = (Mat_<float>(3, 4) << 718.8560, 0., 607.1928, 0., 0., 718.8560, 185.2157, 0., 0,  0., 1., 0.);
    Mat projMatrr = (Mat_<float>(3, 4) << 718.8560, 0., 607.1928, -386.1448, 0., 718.8560, 185.2157, 0., 0,  0., 1., 0.);



    // Mat rotation, translation;
    Mat rotation = Mat::eye(3, 3, CV_64F);
    Mat translation_mono = Mat::zeros(3, 1, CV_64F);
    Mat translation_stereo = Mat::zeros(3, 1, CV_64F);

    Mat pose = Mat::zeros(3, 1, CV_64F);
    Mat Rpose = Mat::eye(3, 3, CV_64F);

    Mat trajectory = Mat::zeros(600, 600, CV_8UC3);

    FeatureSet current_features;

    int init_frame_id = 0;
    Mat image_l, image_r;
    initializeImagesFeatures(init_frame_id, image_l, image_r, current_features);



    for (int frame_id = init_frame_id; frame_id < 1000; frame_id++)
    {

        std::cout << std::endl << "frame_id " << frame_id << std::endl;

        visualOdometry(frame_id, 
                       projMatrl, projMatrr,
                       rotation, translation_mono, translation_stereo, 
                       image_l, image_r,
                       current_features);

        // integrateOdometryMono(frame_id, pose, Rpose, rotation, translation_mono);
        // integrateOdometryScale(frame_id, pose, Rpose, rotation, translation_mono, translation_stereo);
        integrateOdometryStereo(frame_id, pose, Rpose, rotation, translation_stereo);

        std::cout << "Pose" << pose.t() << std::endl;
 
        display(frame_id, trajectory, pose, pose_matrix_gt);


    }

    return 0;
}

