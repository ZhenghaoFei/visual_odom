#ifndef SOLVER_H
#define SOLVER_H

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

void hom2cart2( Eigen::Matrix<double, 3,  Eigen::Dynamic> & points3_l,  Eigen::Matrix<double, 3,  Eigen::Dynamic> & points3_r)
{
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
    std::vector<cv::Point2f> points_l_t1;
    std::vector<cv::Point2f> points_r_t1;
    int function_nums;

    reprojection_error_function(Eigen::Matrix<double, 3, 4> proj_0, 
                                Eigen::Matrix<double, 3, 4> proj_1, 
                                Eigen::Matrix<double, 3, 3> rotation,
                                Eigen::Matrix4d T_left2right,
                                Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> points4D_t0_eigen,
                                std::vector<cv::Point2f> points_left_t1, 
                                std::vector<cv::Point2f> points_right_t1,
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


#endif
