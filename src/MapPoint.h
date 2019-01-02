#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <opencv2/opencv.hpp>
#include "opencv2/features2d/features2d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm>
#include <iterator>
#include <vector>
#include <string>
#include <map>

#include "Frame.h"

struct Observation 
{
    int frame_id;
    cv::Mat pointPoseInFrame;
 };

class MapPoint
 {
 public:
   // MapPoint();
   MapPoint(int id, cv::Mat worldPos);

   ~MapPoint();

   void addObservation(Observation observation);

   int mId;

   // Position in absolute coordinates
   cv::Mat mWorldPos;

   // std::map<Frame*, size_t> mObservations;
   std::vector<Observation> mObservations;
 };


#endif
