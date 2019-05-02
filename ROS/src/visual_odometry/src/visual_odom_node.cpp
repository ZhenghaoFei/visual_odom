
#include <ros/ros.h>
#include "VisualOdometer.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

int main(int argc, char **argv)
{

  ros::init(argc, argv, "visual_odom_node");
  ros::NodeHandle nh;

  VisualOdometer visual_odometer(nh);

  image_transport::ImageTransport it(nh);


  image_transport::SubscriberFilter left_image_sub(it, "/zed/left/image_rect_color", 1);
  image_transport::SubscriberFilter right_image_sub(it, "/zed/right/image_rect_color", 1);
  // image_transport::SubscriberFilter right_cam_info_sub(it, "/zed/right/camera_info", 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> left_cam_info_sub(nh, "/zed/left/camera_info", 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> right_cam_info_sub(nh, "/zed/right/camera_info", 1);



  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> sync(left_image_sub, right_image_sub, left_cam_info_sub, right_cam_info_sub, 10);
  
  sync.registerCallback(boost::bind(&VisualOdometer::imageGrabCallback, &visual_odometer, _1, _2, _3, _4));

  ros::spin();

  return 0;
}