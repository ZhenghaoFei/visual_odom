#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <limits>

#include "matrix.h"



struct errors {
  int32_t first_frame;
  float   r_err;
  float   t_err;
  float   len;
  float   speed;
  errors (int32_t first_frame,float r_err,float t_err,float len,float speed) :
    first_frame(first_frame),r_err(r_err),t_err(t_err),len(len),speed(speed) {}
};


std::vector<Matrix> loadPoses(std::string file_name);

std::vector<float> trajectoryDistances (std::vector<Matrix> &poses);

void plotPathPlot (std::string dir,std::vector<int32_t> &roi,int32_t idx);

void saveErrorPlots(std::vector<errors> &seq_err,std::string plot_error_dir,char* prefix);

void plotErrorPlots (std::string dir,char* prefix);

float rotationError(Matrix &pose_error);

float translationError(Matrix &pose_error);

std::vector<errors> calcSequenceErrors (std::vector<Matrix> &poses_gt,std::vector<Matrix> &poses_result);

void saveSequenceErrors (std::vector<errors> &err,std::string file_name);
