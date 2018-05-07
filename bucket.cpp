
#include "bucket.h"


Bucket::Bucket(int size){
    max_size = size;
}

int Bucket::size(){
    return features.points.size();
}


void Bucket::add_feature(cv::Point2f point, int age){
    if (size()<max_size)
    {
        features.points.push_back(point);
        features.ages.push_back(age);
    }
}