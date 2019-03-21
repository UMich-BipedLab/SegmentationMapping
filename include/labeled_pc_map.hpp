/***********************************
 *
 * ROS_PC_MAP
 *
 ***********************************/
/* Author: Ray Zhang, rzh@umich.edu */

#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <ros/ros.h>
#include <ros/console.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

#include <iostream>
#include <string>
#include <memory>
#include <boost/make_shared.hpp>

#include <eigen_conversions/eigen_msg.h>
#include  <tf_conversions/tf_eigen.h>
#include <tf/transform_datatypes.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

// for the newly defined pointtype
#define PCL_NO_PRECOMPILE
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <boost/shared_ptr.hpp>
#include <pcl/impl/point_types.hpp>

namespace pcl {
  template <unsigned int NUM_CLASS>
  struct PointSegmentedDistribution
  {
    PCL_ADD_POINT4D;                 
    PCL_ADD_RGB;
    int   label;
    float label_distribution[NUM_CLASS];   // templated on any number of classes. TODO
    //float label_distribution[14];
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
  } EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

}

namespace segmentation_projection {
  template<unsigned int NUM_CLASS>
  class PointCloudPainter {
  public:
    PointCloudPainter()
      : nh_()
      , painting_enabled_(true)
      , distribution_enabled_(false)
      , save_pcd_enabled_(false)
      , static_frame_("/map")
      , body_frame_("/body")
      , stacking_visualization_enabled_(true)
      , stacked_pc_ptr_(new pcl::PointCloud<pcl::PointXYZRGB>)
      , path_visualization_enabled_(true)
    {
      // Parse parameters
      ros::NodeHandle pnh("~");
      pnh.getParam("painting_enabled", painting_enabled_);
      pnh.getParam("static_frame", static_frame_);
      pnh.getParam("body_frame", body_frame_);
      pnh.getParam("save_pcd_enabled", save_pcd_enabled_);

      this->pc_subscriber_ = pnh.subscribe("cloud_in", 100, &PointCloudPainter::PointCloudCallback, this);
      this->pc2_subscriber_ = pnh.subscribe("cloud2_in", 100, &PointCloudPainter::PointCloud2Callback, this);
      this->pc_publisher_ = pnh.advertise<sensor_msgs::PointCloud2>("cloud_out", 10);

      if (stacking_visualization_enabled_) {
        this->stacked_pc_publisher_ = pnh.advertise<sensor_msgs::PointCloud2>("stacked_pc_out", 10);
      }

      if (path_visualization_enabled_) {
        this->path_publisher_ = pnh.advertise<nav_msgs::Path>("path_out", 1);
      }
      
      ROS_INFO("ros_pc_map init finish\n");
    }

    void PointCloud2Callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    void PointCloudCallback(const sensor_msgs::PointCloudConstPtr& cloud_msg);


  
  private:
    ros::NodeHandle nh_;
    ros::Subscriber pc_subscriber_;
    ros::Subscriber pc2_subscriber_;
    ros::Publisher pc_publisher_;
    tf::TransformListener listener_;

    std::string static_frame_;
    std::string body_frame_;
    
    bool painting_enabled_;
    bool distribution_enabled_;
    bool save_pcd_enabled_;
    bool stacking_visualization_enabled_;
    bool path_visualization_enabled_;

    // for voxel grid map
    ros::Publisher stacked_pc_publisher_;
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr stacked_pc_ptr_;
    void merge_new_pc_to_voxel_grids(pcl::PointCloud<pcl::PointXYZRGB> & new_cloud, tf::StampedTransform & T_map2body);

    // for path visualization
    ros::Publisher path_publisher_;
    nav_msgs::Path path_;
    void attach_new_pose_to_path(tf::StampedTransform & T_map2body_new, const std_msgs::Header & header);
  };


  template <unsigned int NUM_CLASS>
  inline void
  PointCloudPainter<NUM_CLASS>::merge_new_pc_to_voxel_grids(pcl::PointCloud<pcl::PointXYZRGB> & new_cloud, tf::StampedTransform & T_map2body){
    Eigen::Affine3d T_eigen;
    tf::transformTFToEigen (T_map2body,T_eigen);
    pcl::PointCloud<pcl::PointXYZRGB> transformed_cloud;
    pcl::transformPointCloud (new_cloud, transformed_cloud , T_eigen);
    *stacked_pc_ptr_ = *stacked_pc_ptr_ + transformed_cloud;
    //std::cout<<"# of stacked pc ptr is "<<stacked_pc_ptr_->height * stacked_pc_ptr_->width<<std::endl;
    
    /*
      this->voxel_.setInputCloud (stacked_pc_ptr_);
      this->voxel_.setLeafSize (0.1f, 0.1f, 0.1f);
      this->voxel_.filter (*stacked_pc_ptr_);
    */
    
  }

  template <unsigned int NUM_CLASS>
  inline void
  PointCloudPainter<NUM_CLASS>::attach_new_pose_to_path(tf::StampedTransform & T_map2body_new, const std_msgs::Header & header) {
 
    Eigen::Affine3d T_eigen;
    tf::transformTFToEigen (T_map2body_new,T_eigen);
   
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.frame_id = this->body_frame_;
    tf::poseEigenToMsg (T_eigen, pose_stamped.pose);
    std::cout<<"xyz: "<<T_eigen(0, 3)<<", "<<T_eigen(1, 3)<<", "<<T_eigen(2, 3)<<", at time:"<< header.stamp.toNSec()<<" \n";

    path_.header.stamp = header.stamp;
    path_.header.frame_id = this->static_frame_;
    path_.poses.push_back(pose_stamped);
      
  }
  
  template <unsigned int NUM_CLASS>
  inline void
  PointCloudPainter<NUM_CLASS>::PointCloud2Callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    /*
    sensor_msgs::PointCloud2 painted_cloud = *cloud_msg;
    if (!painting_enabled_) {
      painted_cloud.header.frame_id = "gtsam_imu";
      pc_publisher_.publish(painted_cloud);
    }
  
    // Get rgb img name from point cloud timestamp
    std::string rgb_img_name = std::to_string( long(GetMsgTime(cloud_msg) * 1e6) );
    rgb_img_name = rgb_img_folder_ + rgb_img_name + "000.png";
  
    // Load the corresponding rgb image
    cv::Mat rgb_img = cv::imread(rgb_img_name, CV_LOAD_IMAGE_COLOR);
    cv::Mat img;

    if (!rgb_img.rows) // return if no img found
      return;
    else  // resize to organized point cloud size
      cv::resize(rgb_img, img, cv::Size(cloud_msg->width, cloud_msg->height));

    // Paint point clouds
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(painted_cloud, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(painted_cloud, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(painted_cloud, "b");
  
    for (int j = 0; j < img.rows; ++j) {
      for (int i = 0; i < img.cols; ++i) {

        *iter_r = img.at<cv::Vec3b> (cv::Point(i, j)) [2]; // b
        *iter_g = img.at<cv::Vec3b> (cv::Point(i, j)) [1]; // g
        *iter_b = img.at<cv::Vec3b> (cv::Point(i, j)) [0]; // r
      
        ++iter_r;
        ++iter_g;
        ++iter_b;
      }
    }
    pc_publisher_.publish(painted_cloud);
    */
  }

  template <unsigned int NUM_CLASS>
  inline void
  PointCloudPainter<NUM_CLASS>::PointCloudCallback(const sensor_msgs::PointCloudConstPtr& cloud_msg) {

    sensor_msgs::PointCloud2 painted_cloud;
    sensor_msgs::PointCloud2 stacked_cloud;
    //sensor_msgs::convertPointCloudToPointCloud2(*cloud_msg, painted_cloud);

    pcl::PointCloud<pcl::PointXYZRGB> pointcloud_pcl;
    pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS> > pointcloud_seg;

    tf::StampedTransform transform;
    try{
      this->listener_.waitForTransform(this->static_frame_, this->body_frame_,
                                       cloud_msg->header.stamp, ros::Duration(4.0) );
      
      this->listener_.lookupTransform(this->static_frame_, this->body_frame_,  
                                      cloud_msg->header.stamp, transform);
    }
    catch (tf::TransformException ex){
      std::cout<<"tf look for failed\n";
      ROS_ERROR("%s",ex.what());
      return;
    }
  
    //for (int j = 0; j < img.rows; ++j) {
    //std::cout<<"At time "<<cloud_msg->header.stamp.toSec()<<", # of lidar pts is "<<cloud_msg->points.size()<<std::endl;
    for (int i = 0; i < cloud_msg->points.size(); ++i) {
      pcl::PointXYZRGB p;
      p.x = cloud_msg->points[i].x;
      p.y = cloud_msg->points[i].y;
      p.z = cloud_msg->points[i].z;
      // pack r/g/b into rgb
      uint8_t r = cloud_msg->channels[1].values[i];
      uint8_t g = cloud_msg->channels[2].values[i];
      uint8_t b = cloud_msg->channels[3].values[i];    // Example: Red color
      uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
      p.rgb = *reinterpret_cast<float*>(&rgb);

      pcl::PointSegmentedDistribution<NUM_CLASS> p_seg;
      p_seg.x = cloud_msg->points[i].x;
      p_seg.y = cloud_msg->points[i].y;
      p_seg.z = cloud_msg->points[i].z;
      p_seg.rgb = *reinterpret_cast<float*>(&rgb);
      p_seg.label = cloud_msg->channels[0].values[i];

      if (distribution_enabled_) {
        float sums = 0;
        for (int c = 0; c != NUM_CLASS; c++){
          p_seg.label_distribution[c] = cloud_msg->channels[c+7].values[i];
          sums += p_seg.label_distribution[c];
        }
        assert(sums > 0.99 & sums < 1.01);
      }
    
      pointcloud_pcl.push_back(p);
      pointcloud_seg.push_back(p_seg);
    }

    // publish pointxyzrgb raw at the current timestamp
    pcl::toROSMsg(pointcloud_pcl, painted_cloud);
    painted_cloud.header = cloud_msg->header;
    painted_cloud.header.frame_id = this->body_frame_;
    pc_publisher_.publish(painted_cloud);

    // publish the stacked pc map
    if (this->stacking_visualization_enabled_) {
      this->merge_new_pc_to_voxel_grids(pointcloud_pcl, transform);            
      pcl::toROSMsg(*(this->stacked_pc_ptr_), stacked_cloud);
      stacked_cloud.header = cloud_msg->header;
      stacked_cloud.header.frame_id = this->static_frame_;
      stacked_pc_publisher_.publish((stacked_cloud));
    }


    // publish the path
    if (this->path_visualization_enabled_) {
      this->attach_new_pose_to_path(transform,  cloud_msg->header);
      path_publisher_.publish(this->path_);
    }
    
    // save the result as pcd files
    if (this->save_pcd_enabled_) {
      std::string name_pcd = std::to_string(cloud_msg->header.stamp.toNSec());
      pcl::io::savePCDFile ("segmented_pcd/" + name_pcd + ".pcd", pointcloud_seg);
    }
  }


} 


