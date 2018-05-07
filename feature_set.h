#ifndef FEATURE_SET_H
#define FEATURE_SET_H

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

struct FeatureSet {
    std::vector<cv::Point2f>  points;
    std::vector<int>  ages;
 };

#endif
