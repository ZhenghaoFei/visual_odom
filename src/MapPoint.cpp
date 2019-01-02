#include "MapPoint.h"

MapPoint::MapPoint(int id, cv::Mat worldPos)
{
    mId = id;
    mWorldPos = worldPos;
}


MapPoint::~MapPoint()
{}

void MapPoint::addObservation(Observation observation)
{
    mObservations.push_back(observation);
}