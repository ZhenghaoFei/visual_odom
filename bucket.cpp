
#include "bucket.h"


Bucket::Bucket(int size){
    max_size = size;
}

int Bucket::size(){
    return features.points.size();
}


void Bucket::add_feature(cv::Point2f point, int age){
    // won't add feature with age > 10;
    int age_threshold = 10;
    if (age < age_threshold)
    {
        // insert any feature before bucket is full
        if (size()<max_size)
        {
            features.points.push_back(point);
            features.ages.push_back(age);

        }
        else
        // insert feature with old age and remove youngest one
        {
            int age_min = features.ages[0];
            int age_min_idx = 0;

            for (int i = 0; i < size(); i++)
            {
                if (age < age_min)
                {
                    age_min = age;
                    age_min_idx = i;
                }
            }
            features.points[age_min_idx] = point;
            features.ages[age_min_idx] = age;
        }
    }

}

void Bucket::get_features(FeatureSet& current_features){

    current_features.points.insert(current_features.points.end(), features.points.begin(), features.points.end());
    current_features.ages.insert(current_features.ages.end(), features.ages.begin(), features.ages.end());
}

Bucket::~Bucket(){
}