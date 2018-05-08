#include "feature.h"
#include "bucket.h"

void deleteUnmatchFeatures(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1, std::vector<uchar>& status){
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  int indexCorrection = 0;
  for( int i=0; i<status.size(); i++)
     {  cv::Point2f pt = points1.at(i- indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))   
        {
              if((pt.x<0)||(pt.y<0))    
              {
                status.at(i) = 0;
              }
              points0.erase (points0.begin() + (i - indexCorrection));
              points1.erase (points1.begin() + (i - indexCorrection));
              indexCorrection++;
        }
     }
}

void featureDetectionFast(cv::Mat image, std::vector<cv::Point2f>& points)  {   //uses FAST as of now, modify parameters as necessary
  std::vector<cv::KeyPoint> keypoints;
  int fast_threshold = 20;
  bool nonmaxSuppression = true;
  cv::FAST(image, keypoints, fast_threshold, nonmaxSuppression);
  cv::KeyPoint::convert(keypoints, points, std::vector<int>());
}

void featureDetectionGoodFeaturesToTrack(cv::Mat image, std::vector<cv::Point2f>& points)  {   //uses FAST as of now, modify parameters as necessary

  int maxCorners = 5000;
  double qualityLevel = 0.01;
  double minDistance = 5.;
  int blockSize = 3;
  bool useHarrisDetector = false;
  double k = 0.04;
  cv::Mat mask;

  cv::goodFeaturesToTrack( image, points, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );
}

void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2, std::vector<uchar>& status)   { 
//this function automatically gets rid of points for which tracking fails

  std::vector<float> err;                    
  cv::Size winSize=cv::Size(21,21);                                                                                             
  cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

  calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
  deleteUnmatchFeatures(points1, points2, status);

}

void deleteUnmatchFeaturesCircle(std::vector<cv::Point2f>& points0, std::vector<cv::Point2f>& points1,
                          std::vector<cv::Point2f>& points2, std::vector<cv::Point2f>& points3,
                          std::vector<uchar>& status0, std::vector<uchar>& status1,
                          std::vector<uchar>& status2, std::vector<uchar>& status3,
                          std::vector<int>& ages){
  //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
  for (int i = 0; i < ages.size(); ++i)
  {
     ages[i] += 1;
  }

  int indexCorrection = 0;
  for( int i=0; i<status3.size(); i++)
     {  cv::Point2f pt0 = points0.at(i- indexCorrection);
        cv::Point2f pt1 = points1.at(i- indexCorrection);
        cv::Point2f pt2 = points2.at(i- indexCorrection);
        cv::Point2f pt3 = points3.at(i- indexCorrection);
        
        if ((status3.at(i) == 0)||(pt3.x<0)||(pt3.y<0)||
            (status2.at(i) == 0)||(pt2.x<0)||(pt2.y<0)||
            (status1.at(i) == 0)||(pt1.x<0)||(pt1.y<0)||
            (status0.at(i) == 0)||(pt0.x<0)||(pt0.y<0))   
        {
          if((pt0.x<0)||(pt0.y<0)||(pt1.x<0)||(pt1.y<0)||(pt2.x<0)||(pt2.y<0)||(pt3.x<0)||(pt3.y<0))    
          {
            status3.at(i) = 0;
          }
          points0.erase (points0.begin() + (i - indexCorrection));
          points1.erase (points1.begin() + (i - indexCorrection));
          points2.erase (points2.begin() + (i - indexCorrection));
          points3.erase (points3.begin() + (i - indexCorrection));
          ages.erase (ages.begin() + (i - indexCorrection));
          indexCorrection++;
        }

     }  
}

void circularMatching(cv::Mat img_l_0, cv::Mat img_r_0, cv::Mat img_l_1, cv::Mat img_r_1,
                      std::vector<cv::Point2f>& points_l_0, std::vector<cv::Point2f>& points_r_0,
                      std::vector<cv::Point2f>& points_l_1, std::vector<cv::Point2f>& points_r_1,
                      FeatureSet& current_features) { 
  
  //this function automatically gets rid of points for which tracking fails

  std::vector<float> err;                    
  cv::Size winSize=cv::Size(21,21);                                                                                             
  cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);

  std::vector<uchar> status0;
  std::vector<uchar> status1;
  std::vector<uchar> status2;
  std::vector<uchar> status3;


  calcOpticalFlowPyrLK(img_l_0, img_r_0, points_l_0, points_r_0, status0, err, winSize, 3, termcrit, 0, 0.001);
  calcOpticalFlowPyrLK(img_r_0, img_r_1, points_r_0, points_r_1, status1, err, winSize, 3, termcrit, 0, 0.001);
  calcOpticalFlowPyrLK(img_r_1, img_l_1, points_r_1, points_l_1, status2, err, winSize, 3, termcrit, 0, 0.001);
  calcOpticalFlowPyrLK(img_l_1, img_l_0, points_l_1, points_l_0, status3, err, winSize, 3, termcrit, 0, 0.001);
  
  deleteUnmatchFeaturesCircle(points_l_0, points_r_0, points_r_1, points_l_1,
                        status0, status1, status2, status3, current_features.ages);

  // std::cout << "points : " << points_l_0.size() << " "<< points_r_0.size() << " "<< points_r_1.size() << " "<< points_l_1.size() << " "<<std::endl;
}


void bucketingFeatures(cv::Mat& image, FeatureSet& current_features, int bucket_size, int features_per_bucket){
// This function buckets features
// image: only use for getting dimension of the image
// bucket_size: bucket size in pixel is bucket_size*bucket_size
// features_per_bucket: number of selected features per bucket
    int image_height = image.rows;
    int image_width = image.cols;

    int buckets_nums_height = image_height/bucket_size;
    int buckets_nums_width = image_width/bucket_size;

    int buckets_number = buckets_nums_height * buckets_nums_width;

    // std::cout << "image_height " << image_height << std::endl;
    // std::cout << "image_width " << image_width << std::endl;
    // std::cout << "buckets_nums_height " << buckets_nums_height << std::endl;
    // std::cout << "buckets_nums_width " << buckets_nums_width << std::endl;

    std::vector<Bucket> Buckets;

    // initialize all the buckets
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
      {
        Buckets.push_back(Bucket(features_per_bucket));
      }
    }
    // std::cout << "buckets number " << Buckets.size() << std::endl;

    // bucket all current features into buckets by their location
    int buckets_nums_height_idx, buckets_nums_width_idx, buckets_idx;
    for (int i = 0; i < current_features.points.size(); ++i)
    {
      buckets_nums_height_idx = current_features.points[i].y/bucket_size;
      buckets_nums_width_idx = current_features.points[i].x/bucket_size;
      // std::cout << "buckets_nums_height_idx: " << buckets_nums_height_idx << ", " << buckets_nums_width_idx << std::endl;

      buckets_idx = buckets_nums_height_idx*buckets_nums_width + buckets_nums_width_idx;
      Buckets[buckets_idx].add_feature(current_features.points[i], current_features.ages[i]);

    }

    // get features back from buckets
    current_features.clear();
    for (int buckets_idx_height = 0; buckets_idx_height <= buckets_nums_height; buckets_idx_height++)
    {
      for (int buckets_idx_width = 0; buckets_idx_width <= buckets_nums_width; buckets_idx_width++)
      {
         buckets_idx = buckets_idx_height*buckets_nums_width + buckets_idx_width;
         // std::cout << "buckets " << buckets_idx << " size " << Buckets[buckets_idx].size() << std::endl;
         Buckets[buckets_idx].get_features(current_features);
      }
    }

    std::cout << "current features number after bucketing: " << current_features.size() << std::endl;

}

void appendNewFeatures(cv::Mat& image, FeatureSet& current_features){
    std::vector<cv::Point2f>  points_new;
    featureDetectionGoodFeaturesToTrack(image, points_new);
    current_features.points.insert(current_features.points.end(), points_new.begin(), points_new.end());
    std::vector<int>  ages_new(points_new.size(), 0);
    current_features.ages.insert(current_features.ages.end(), ages_new.begin(), ages_new.end());
}
