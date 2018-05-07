
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

void Bucket::get_features(FeatureSet& current_features){

    current_features.points.insert(current_features.points.end(), features.points.begin(), features.points.end());
    current_features.ages.insert(current_features.ages.end(), features.ages.begin(), features.ages.end());
}

Bucket::~Bucket(){
}