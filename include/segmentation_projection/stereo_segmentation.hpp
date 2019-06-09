#pragma once
// for the PointSegmentedDistribution
#include "PointSegmentedDistribution.hpp"

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>



namespace segmentation_projection {

  typedef message_filters::sync_policies::ApproximateTime
    <sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> sync_pol;

  template <unsigned int NUM_CLASS>
  class StereoSegmentation {
    public:
      StereoSegmentation()
        : nh_() {
          ros::NodeHandle pnh("~");
          // Subscribe messages
          depth_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (pnh, "/camera/aligned_depth_to_color/image_raw", 1);
          color_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (pnh, "/camera/color/image_raw", 1);
          depth_cam_sub_ = new message_filters::Subscriber<sensor_msgs::CameraInfo> (pnh, "/camera/aligned_depth_to_color/camera_info", 1);
          sync_ = new message_filters::Synchronizer<sync_pol> (sync_pol(10), *depth_sub_, *color_sub_, *depth_cam_sub_);
          sync_->registerCallback(boost::bind(&StereoSegmentation::DepthColorCallback, this, _1, _2, _3));
          
          // Publish messages
          pc1_publisher_ = pnh.advertise<sensor_msgs::PointCloud>("cloud_out1", 1);
          pc2_publisher_ = pnh.advertise<sensor_msgs::PointCloud2>("cloud_out2", 1);
        }

     void DepthColorCallback(const sensor_msgs::ImageConstPtr& depth_msg,
                             const sensor_msgs::ImageConstPtr& color_msg,
                             const sensor_msgs::CameraInfoConstPtr& camera_info_msg);
     void Depth2PointCloud1(const sensor_msgs::ImageConstPtr& depth_msg,
                            const sensor_msgs::ImageConstPtr& color_msg);
     void Depth2PointCloud2(const sensor_msgs::ImageConstPtr& depth_msg,
                            const sensor_msgs::ImageConstPtr& color_msg);
    
    private:
      ros::NodeHandle nh_;
      message_filters::Subscriber<sensor_msgs::Image> *depth_sub_;
      message_filters::Subscriber<sensor_msgs::Image>* color_sub_;
      message_filters::Subscriber<sensor_msgs::CameraInfo>* depth_cam_sub_;
      message_filters::Synchronizer<sync_pol>* sync_;
      ros::Publisher pc1_publisher_;
      ros::Publisher pc2_publisher_;
      // For camera depth to point cloud
      image_geometry::PinholeCameraModel model_;

  };

  template <unsigned int NUM_CLASS>
  inline void
  StereoSegmentation<NUM_CLASS>::DepthColorCallback(const sensor_msgs::ImageConstPtr& depth_msg,
                                                    const sensor_msgs::ImageConstPtr& color_msg,
                                                    const sensor_msgs::CameraInfoConstPtr& camera_info_msg) { 
    // Check for bad inputs
    if (depth_msg->header.frame_id != color_msg->header.frame_id) {
      ROS_ERROR("Depth iamge frame id [%s] doesn't match color image frame id [%s]",
          depth_msg->header.frame_id.c_str(), color_msg->header.frame_id.c_str());
      return;
    }
    
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

    // Update camera model
    this->model_.fromCameraInfo(camera_info_msg);

    Depth2PointCloud1(depth_msg, color_msg);
    Depth2PointCloud2(depth_msg, color_msg);

  };

  template <unsigned int NUM_CLASS>
  inline void
  StereoSegmentation<NUM_CLASS>::Depth2PointCloud1(const sensor_msgs::ImageConstPtr& depth_msg,
                                                   const sensor_msgs::ImageConstPtr& color_msg) {
    
    // Set up to-publish cloud msg
    sensor_msgs::PointCloud::Ptr cloud_msg(new sensor_msgs::PointCloud);
    cloud_msg->header = depth_msg->header;  // use depth image time stamp
    sensor_msgs::ChannelFloat32 label_channel;
    label_channel.name = "labels";
    sensor_msgs::ChannelFloat32 r_channel;
    r_channel.name = "r";
    sensor_msgs::ChannelFloat32 g_channel;
    g_channel.name = "g";
    sensor_msgs::ChannelFloat32 b_channel;
    b_channel.name = "b";

    sensor_msgs::ChannelFloat32 distribution_channel;

    // Use correct principal point from camera info
    float center_x = model_.cx();
    float center_y = model_.cy();

    // Combine unit conversion with scaling by focal length for computing X, Y
    double unit_scaling = 0.001f;
    float constant_x = unit_scaling / model_.fx();
    float constant_y = unit_scaling / model_.fy();
    float bad_point = std::numeric_limits<float>::quiet_NaN();
  
    const uint16_t* depth_row = reinterpret_cast<const uint16_t*>(&depth_msg->data[0]);
    int row_step = depth_msg->step / sizeof(uint16_t);
    const uint8_t* color = &color_msg->data[0];
    int color_step = 3;  // 1 if mono
    int color_skip = color_msg->step - color_msg->width * color_step;

    // Iterate through depth image
    int i = 0;  // num of point
    for (int v = 0; v < int(depth_msg->height); ++v, depth_row += row_step, color += color_skip){
      for (int u = 0; u < int(depth_msg->width); ++u, color += color_step, ++i) {
      uint16_t depth = depth_row[u];

      // Check for invalid measurements
      geometry_msgs::Point32 p;
      if (depth <= 0) {
        p.x = p.y = p.z = bad_point;
      }
      else {
        // Fill in XYZ
        p.x = (u - center_x) * depth * constant_x;
        p.y = (v - center_y) * depth * constant_y;
        p.z = (float) depth * unit_scaling;
      }

      cloud_msg->points.push_back(p);

      // Fill in r g b channels: bgr
      r_channel.values.push_back(color[0]);
      g_channel.values.push_back(color[1]);
      b_channel.values.push_back(color[2]);

      // Fill in semantics
      label_channel.values.push_back(1);
      distribution_channel.values.push_back(1);


    }
   }
    
    cloud_msg->channels.push_back(label_channel);
    cloud_msg->channels.push_back(r_channel);
    cloud_msg->channels.push_back(g_channel);
    cloud_msg->channels.push_back(b_channel);
    cloud_msg->channels.push_back(r_channel);
    cloud_msg->channels.push_back(g_channel);
    cloud_msg->channels.push_back(b_channel);
    cloud_msg->channels.push_back(distribution_channel);

    
    pc1_publisher_.publish(cloud_msg);
  };

  
  
  template <unsigned int NUM_CLASS>
  inline void
  StereoSegmentation<NUM_CLASS>::Depth2PointCloud2(const sensor_msgs::ImageConstPtr& depth_msg,
                                                   const sensor_msgs::ImageConstPtr& color_msg) {
   // Set up to-publish cloud msg
   sensor_msgs::PointCloud2::Ptr cloud_msg (new sensor_msgs::PointCloud2);
   cloud_msg->header = depth_msg->header;  // use depth image time stamp
   cloud_msg->height = depth_msg->height;
   cloud_msg->width = depth_msg->width;
   cloud_msg->is_dense = false;
   cloud_msg->is_bigendian = false;

   sensor_msgs::PointCloud2Modifier pcd_modifier(*cloud_msg);
   pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

   // Use correct principal point from camera info
   float center_x = model_.cx();
   float center_y = model_.cy();

   // Combine unit conversion with scaling by focal length for computing X,Y
   double unit_scaling = 0.001f;
   float constant_x = unit_scaling / model_.fx();
   float constant_y = unit_scaling / model_.fy();
   float bad_point = std::numeric_limits<float>::quiet_NaN();

   const uint16_t* depth_row = reinterpret_cast<const uint16_t*>(&depth_msg->data[0]);
   int row_step = depth_msg->step / sizeof(uint16_t);
   const uint8_t* color = &color_msg->data[0];
   int color_step = 3;  // 1 if mono
   int color_skip = color_msg->step - color_msg->width * color_step;


   sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_msg, "x");
   sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_msg, "y");
   sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_msg, "z");
   sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*cloud_msg, "r");
   sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*cloud_msg, "g");
   sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*cloud_msg, "b");
   sensor_msgs::PointCloud2Iterator<uint8_t> iter_a(*cloud_msg, "a");

   // Iterate through depth image
   for (int v = 0; v < int(cloud_msg->height); ++v, depth_row += row_step, color += color_skip){
    for (int u = 0; u < int(cloud_msg->width); ++u, color += color_step, 
        ++iter_x, ++iter_y, ++iter_z, ++iter_a, ++iter_r, ++iter_g, ++iter_b) {
      uint16_t depth = depth_row[u];

      // Check for invalid measurements
      if (depth <= 0)
        *iter_x = *iter_y = *iter_z = bad_point;
      else {
        // Fill in XYZ
        *iter_x = (u - center_x) * depth * constant_x;
        *iter_y = (v - center_y) * depth * constant_y;
        *iter_z = (float) depth * unit_scaling;
      }
      
      if (*iter_z > 10)
        *iter_z = bad_point;

      // Fill in color
      *iter_a = 255;
      *iter_r = color[0];
      *iter_g = color[1];
      *iter_b = color[2];
    
    
    }
   }
    pc2_publisher_.publish(cloud_msg);
  };


}


