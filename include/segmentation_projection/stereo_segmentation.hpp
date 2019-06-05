#pragma once
// for the PointSegmentedDistribution
#include "PointSegmentedDistribution.hpp"

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>




namespace segmentation_projection {

  typedef message_filters::sync_policies::ApproximateTime
    <sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image, sensor_msgs::CameraInfo> sync_pol;

  template <unsigned int NUM_CLASS>
  class StereoSegmentation {
    public:
      StereoSegmentation()
        : nh_() {
          ros::NodeHandle pnh("~");
          color_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (pnh, "/camera/color/image_raw", 1);
          color_cam_sub_ = new message_filters::Subscriber<sensor_msgs::CameraInfo> (pnh, "/camera/color/camera_info", 1);
          depth_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (pnh, "/camera/aligned_depth_to_color/image_raw", 1);
          depth_cam_sub_ = new message_filters::Subscriber<sensor_msgs::CameraInfo> (pnh, "/camera/aligned_depth_to_color/camera_info", 1);
          sync_ = new message_filters::Synchronizer<sync_pol> (sync_pol(10), *color_sub_, *color_cam_sub_, *depth_sub_, *depth_cam_sub_);
          sync_->registerCallback(boost::bind(&StereoSegmentation::ColorDepthCallback, this,_1, _2, _3, _4));
        }

     void ColorDepthCallback(const sensor_msgs::ImageConstPtr& color_msg,
                                  const sensor_msgs::CameraInfoConstPtr& color_camera_info,
                                  const sensor_msgs::ImageConstPtr& depth_msg,
                                  const sensor_msgs::CameraInfoConstPtr& depth_camera_info);

    private:
      ros::NodeHandle nh_;
      message_filters::Subscriber<sensor_msgs::Image>* color_sub_;
      message_filters::Subscriber<sensor_msgs::CameraInfo>* color_cam_sub_;
      message_filters::Subscriber<sensor_msgs::Image> *depth_sub_;
      message_filters::Subscriber<sensor_msgs::CameraInfo>* depth_cam_sub_;
      message_filters::Synchronizer<sync_pol>* sync_;

  };

  template <unsigned int NUM_CLASS>
  inline void
  StereoSegmentation<NUM_CLASS>::ColorDepthCallback(const sensor_msgs::ImageConstPtr& color_msg,
                                                    const sensor_msgs::CameraInfoConstPtr& color_camera_info,
                                                    const sensor_msgs::ImageConstPtr& depth_msg,
                                                    const sensor_msgs::CameraInfoConstPtr& depth_camera_info) {
    // Get rgb and depth images
    cv_bridge::CvImagePtr color_ptr;
    cv_bridge::CvImagePtr depth_ptr;
    try{
      color_ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::BGR8);
      depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat color = color_ptr->image;
    cv::Mat depth = depth_ptr->image;
  };






}
