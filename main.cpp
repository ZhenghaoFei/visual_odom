
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

#include "evaluate_odometry.h"

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

struct reprojection_error_function : functor<double>
// Reprojection error function to be minimized for translation t
// K1, K2: 3 x 3 Intrinsic parameters matrix for the left and right cameras
// R: 3x3 Estimated rotation matrix from the previous step
// points3D: 3xM 3D Point cloud generated from stereo pair in left camera
// frame
// pts_l: matched feature points locations in left camera frame
// pts_r: matched feature points locations in right camera frame
{
    Eigen::MatrixXd K1;
    int const_a;

    reprojection_error_function(int a): functor<double>(3,3) 
    {
         const_a = a;
    }

    int operator()(const Eigen::VectorXd &x, Eigen::VectorXd &fvec) const
    {
        // Implement y = 10*(x0+3)^2 + (x1-5)^2
        fvec(0) = sqrt(10.0) * (x(0)+3.0);
        fvec(1) = x(1)-5.0;
        fvec(2) = x(2) + const_a;

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

  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  Point2f pt = points2.at(i- indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))    {
              if((pt.x<0)||(pt.y<0))    {
                status.at(i) = 0;
              }
              points1.erase (points1.begin() + (i - indexCorrection));
              points2.erase (points2.begin() + (i - indexCorrection));
              indexCorrection++;
        }

     }
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



void visualOdometry(int current_frame_id, Mat& rotation, Mat& translation, std::vector<Point2f>& current_feature_set)
{

    // ------------
    // load images
    // ------------
    Mat image_left_t0 = loadImageLeft(current_frame_id);
    Mat image_right_t0 = loadImageRight(current_frame_id);

    Mat image_left_t1 = loadImageLeft(current_frame_id + 1);
    Mat image_right_t1 = loadImageRight(current_frame_id + 1);

    // ------------
    // feature detection using FAST
    // ------------
    std::vector<Point2f> points_left_t0, points_right_t0, points_left_t1, points_right_t1;        //vectors to store the coordinates of the feature points
    
    featureDetection(image_left_t0, points_left_t0);        
    featureDetection(image_right_t0, points_right_t0);  

    featureDetection(image_left_t1, points_left_t1);        
    featureDetection(image_right_t1, points_right_t1);     

    if (current_feature_set.size() <= 2000)
    {
        std::cout << "reinitialize feature set: "  << std::endl;
        current_feature_set = points_left_t0;
    }   
    std::cout << "current feature set size: " << current_feature_set.size() << std::endl;

    // ------------
    // feature tracking using KLT tracker and circular matching
    // ------------
    std::vector<uchar> status;
    featureTracking(image_left_t0, image_right_t0, current_feature_set, points_right_t0, status);
    featureTracking(image_right_t0, image_right_t1, points_right_t0, points_right_t1, status);
    featureTracking(image_right_t1, image_left_t1, points_right_t1, points_left_t1, status);
    featureTracking(image_left_t1, image_left_t0, points_left_t1, points_left_t0, status);

    current_feature_set = points_left_t0;
    std::cout << "current feature set size: " << current_feature_set.size() << std::endl;


    // ------------
    // Rotation(R) estimation using Nister's Five Points Algorithm
    // ------------

    //TODO: add a fucntion to load these values directly from KITTI's calib files
    // WARNING: different sequences in the KITTI VO dataset have different intrinsic/extrinsic parameters
    double focal = 718.8560;
    cv::Point2d principle_point(607.1928, 185.2157);

    //recovering the pose and the essential matrix
    Mat E, mask;
    E = findEssentialMat(points_left_t1, points_left_t0, focal, principle_point, RANSAC, 0.999, 1.0, mask);
    recoverPose(E, points_left_t1, points_left_t0, rotation, translation, focal, principle_point, mask);


    // ------------
    // Translation (t) estimation by minimizing the image projection error
    // ------------
    Mat points4D_t0;

    featureTracking(image_left_t0, image_right_t0, points_left_t0, points_right_t0, status);

    Mat projMatr1 = (Mat_<float>(3, 4) << 718.8560, 0., 607.1928, 0., 0., 718.8560, 185.2157, 0., 0., 1., 0.);
    Mat projMatr2 = (Mat_<float>(3, 4) << 718.8560, 0., 607.1928, -386.1448, 0., 718.8560, 185.2157, 0., 0., 1., 0.);

    // std::cout << "points_left_t0: " << points_left_t0.size() << std::endl;
    // std::cout << "points_right_t0: " << points_right_t0.size() << std::endl;

    triangulatePoints( projMatr1,  projMatr2,  points_left_t0,  points_right_t0,  points4D_t0);

    std::cout << "points4D_t0: " << points4D_t0.size() << std::endl;












    // imshow( "Left camera", image_left_t0 );
    // imshow( "Right camera", image_right_t0 );


    drawFeaturePoints(image_left_t0, current_feature_set);
    imshow("points ", image_left_t0 );




    // //-- Draw keypoints
    // Mat img_keypoints_right_t0;

    // std::cout << "keypoints size " << keypoints_right_t0.size() << std::endl;

    // drawKeypoints( image_right_t0, keypoints_right_t0, img_keypoints_right_t0, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    // //-- Show detected (drawn) keypoints
    // imshow("Keypoints 1", img_keypoints_right_t0 );


    // waitKey(0);
    // return 0;



    // if ( !image.data )
    // {
    //     printf("No image data \n");
    //     return -1;
    // }
    // namedWindow("Display Image", WINDOW_AUTOSIZE );
    // imshow("Display Image", image);

    // waitKey(0);

}

void integrateOdometryMono(int frame_id, Mat& pose, Mat& Rpose, const Mat& rotation, const Mat& translation)
{
    double scale = 1.00;
    scale = getAbsoluteScale(frame_id);

    if ((scale>0.1)&&(translation.at<double>(2) > translation.at<double>(0)) && (translation.at<double>(2) > translation.at<double>(1))) {

        pose = pose + scale * Rpose * translation;
        Rpose = rotation * Rpose;
    }
    
    else {
     std::cout << "[WARNING] scale below 0.1, or incorrect translation" << std::endl;
    }


}

void integrateOdometry(int frame_id, Mat& pose, Mat& Rpose, const Mat& rotation, const Mat& translation)
{

    pose = pose +  Rpose * translation;
    Rpose = rotation * Rpose;

}

void display(Mat& trajectory, Mat& pose, Mat& pose_gt)
{
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

    // Mat rotation, translation;
    Mat rotation = Mat::eye(3, 3, CV_64F);
    Mat translation = Mat::zeros(3, 1, CV_64F);
    Mat pose = Mat::zeros(3, 1, CV_64F);
    Mat Rpose = Mat::eye(3, 3, CV_64F);
    Mat pose_gt = Mat::zeros(1, 3, CV_64F);

    Mat trajectory = Mat::zeros(600, 600, CV_8UC3);
    std::vector<Point2f> current_feature_set;

    for (int frame_id = 0; frame_id < 1000; frame_id++)
    {

        std::cout << "frame_id " << frame_id << std::endl;
        // std::cout << "current feature set size: " << current_feature_set.size() << std::endl;

        visualOdometry(frame_id, rotation, translation, current_feature_set);
        integrateOdometryMono(frame_id, pose, Rpose, rotation, translation);
        integrateOdometry(frame_id, pose, Rpose, rotation, translation);

        // std::cout << "R" << std::endl;
        // std::cout << rotation << std::endl;
        // std::cout << "t" << std::endl;
        // std::cout << translation << std::endl;
        std::cout << "Pose" << std::endl;
        std::cout << pose << std::endl;

        pose_gt.at<double>(0) = pose_matrix_gt[frame_id].val[0][3];
        pose_gt.at<double>(1) = pose_matrix_gt[frame_id].val[0][7];
        pose_gt.at<double>(2) = pose_matrix_gt[frame_id].val[0][11];
 
        display(trajectory, pose, pose_gt);


    }

    return 0;
}

