#include "Frame.h"

Frame::Frame()
{}


Frame::Frame(int frameId, const cv::Mat projMatL, const cv::Mat projMatR, cv::Mat worldRotation, cv::Mat worldTranslation)
{
    m_projMatL = projMatL;
    m_projMatR = projMatR;
    m_worldRotation = worldRotation;
    m_worldTranslation = worldTranslation;

}



void Frame::setFeatures(std::vector<cv::Point2f> pointsFeatureLeft, std::vector<cv::Point2f> pointsFeatureRight)
{
    m_pointsFeatureLeft = pointsFeatureLeft;
    m_pointsFeatureRight = pointsFeatureRight;

}

void Frame::triangulateFeaturePoints(cv::Mat& points4D)
{
    cv::triangulatePoints( m_projMatL,  m_projMatR,  m_pointsFeatureLeft,  m_pointsFeatureRight,  points4D);
}


