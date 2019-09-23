#pragma once

// std
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cassert>
#include <tuple>
#include <unordered_map>
#include <math.h>
// ros
#include <ros/ros.h>
#include <ros/console.h>
#include <ros/time.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>

#include <Eigen/Dense>
#include <Eigen/Core>

// opencv
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/eigen.hpp>

// boost
#include <boost/shared_ptr.hpp>

// tensorflow
//#include <tensorflow/c/c_api.h>

// for the PointSegmentedDistribution class
#include "PointSegmentedDistribution.hpp"
#include "SegmentationMapping/ImageLabelDistribution.h"
//#include "tf_inference.hpp"


//namespace tf = tensorflow;
//namespace tf_ops = tensorflow::ops;

namespace SegmentationMapping {


  typedef message_filters::sync_policies::ExactTime
  <sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image, ImageLabelDistribution> sync_pol;

  template <unsigned int NUM_CLASS>
  class StereoSegmentation {
  public:
    StereoSegmentation()
      : nh_()
      , color_topic_("/camera/color/image_raw")
      , depth_topic_("/camera/aligned_depth_to_color/image_raw")
      , depth_cam_topic_("/camera/aligned_depth_to_color/camera_info")
      , label_topic_("/labeled_image")
      , distribution_topic_("/distribution_image")
    {
      ros::NodeHandle pnh("~");
      pnh.getParam("color_topic", color_topic_);
      pnh.getParam("depth_topic", depth_topic_);
      pnh.getParam("depth_cam_topic", depth_cam_topic_);
      pnh.getParam("label_topic", label_topic_);
      pnh.getParam("distribution_topic", distribution_topic_);

      // init the message filter for image, depth, camera info
      color_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (pnh, color_topic_, 50);
      depth_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (pnh, depth_topic_, 50);
      depth_cam_sub_ = new message_filters::Subscriber<sensor_msgs::CameraInfo> (pnh, depth_cam_topic_, 50);
      label_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (pnh, label_topic_, 5);
      distribution_sub_ = new message_filters::Subscriber<ImageLabelDistribution> (pnh, distribution_topic_, 5);
      sync_ = new message_filters::Synchronizer<sync_pol> (sync_pol(300), *depth_sub_, *color_sub_, *depth_cam_sub_, *label_sub_, *distribution_sub_);
      
      sync_->registerCallback(boost::bind(&StereoSegmentation::DepthColorCallback, this,_1, _2, _3, _4, _5));
      
      pc1_publisher_ = pnh.advertise<sensor_msgs::PointCloud>("/labeled_pointcloud", 1);
      pc2_publisher_ = pnh.advertise<sensor_msgs::PointCloud2>("/labeled_pointcloud_color_pc2", 1);


      num_skip_frames = pnh.getParam("skip_every_k_frame", num_skip_frames);

      /*
      // init tensorflow
      std::string input_tensor_name, output_distribution_tensor_name, output_label_tensor_name;
      pnh.getParam("tf_input_tensor", input_tensor_name);
      //pnh.getParam("tf_label_output_tensor", output_label_tensor_name);
      pnh.getParam("tf_distribution_output_tensor", output_distribution_tensor_name);
      std::string frozen_graph_path;
      pnh.getParam("tf_frozen_graph_path", frozen_graph_path );
      ROS_DEBUG("Init tf_Inference....");
      tf_infer.reset(new tfInference(frozen_graph_path, input_tensor_name,
                                   output_label_tensor_name, output_distribution_tensor_name));
      ROS_DEBUG_STREAM("Init tensorflow and revoke frozen graph from "<<frozen_graph_path);
      */
      label2color[2]  =std::make_tuple(250, 250, 250 ); // road
      label2color[3]  =std::make_tuple(250, 250, 250 ); // sidewalk
      label2color[5]  =std::make_tuple(250, 128, 0   ); // building
      label2color[10] =std::make_tuple(192, 192, 192 ); // pole
      label2color[12] =std::make_tuple(250, 250, 0   ); // sign
      label2color[6]  =std::make_tuple(0  , 100, 0   ); // vegetation
      label2color[4]  =std::make_tuple(128, 128, 0   ); // terrain
      label2color[13] =std::make_tuple(135, 206, 235 ); // sky
      label2color[1]  =std::make_tuple( 30, 144, 250 ); // water
      label2color[8]  =std::make_tuple(220, 20,  60  ); // person
      label2color[7]  =std::make_tuple( 0, 0,142     ); // car
      label2color[9]  =std::make_tuple(119, 11, 32   ); // bike
      label2color[11] =std::make_tuple(123, 104, 238 ); // stair
      label2color[0]  =std::make_tuple(255, 255, 255 ); // background
      
    }
    
    ~StereoSegmentation() {
      delete sync_;
      delete color_sub_;
      delete depth_sub_;
      delete depth_cam_sub_;
    }

    void DepthColorCallback(const sensor_msgs::ImageConstPtr& depth_msg,
                            const sensor_msgs::ImageConstPtr& color_msg,
                            const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
                            const sensor_msgs::ImageConstPtr& labeled_msg,
                            const ImageLabelDistributionConstPtr & distribution_msg);

    void Depth2PointCloud1(const sensor_msgs::ImageConstPtr& depth_msg,
                           const sensor_msgs::ImageConstPtr& color_msg,
                           bool publish_semantic,
                           const cv::Mat & label_img,
                           const cv::Mat & distribution_img);
    void Depth2PointCloud2(const sensor_msgs::ImageConstPtr& depth_msg,
                           const sensor_msgs::ImageConstPtr& color_msg,
                           bool publish_semantic,
                           const cv::Mat & label_img);

    
  private:
    ros::NodeHandle nh_;
    std::string color_topic_;
    std::string depth_topic_;
    std::string depth_cam_topic_;
    std::string label_topic_;
    std::string distribution_topic_;

    message_filters::Subscriber<sensor_msgs::Image>* color_sub_;
    message_filters::Subscriber<sensor_msgs::Image> *depth_sub_;
    message_filters::Subscriber<sensor_msgs::CameraInfo>* depth_cam_sub_;
    message_filters::Subscriber<sensor_msgs::Image> *label_sub_;
    message_filters::Subscriber<ImageLabelDistribution> *distribution_sub_;
    message_filters::Synchronizer<sync_pol>* sync_;
    //message_filters::TimeSynchronizer<sync_pol>* sync_;
    //sync_pol* sync_;

    ros::Publisher pc1_publisher_;
    ros::Publisher pc2_publisher_;
    // For camera depth to point cloud
    image_geometry::PinholeCameraModel model_;
    ros::Subscriber label_sub;

    int num_skip_frames;
    //std::unique_ptr<tfInference> tf_infer;
    std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> label2color;
  };
  
  template <unsigned int NUM_CLASS>
  inline void
  StereoSegmentation<NUM_CLASS>::DepthColorCallback(const sensor_msgs::ImageConstPtr& depth_msg,
                                                    const sensor_msgs::ImageConstPtr& color_msg,
                                                    const sensor_msgs::CameraInfoConstPtr& camera_info_msg,
                                                    const sensor_msgs::ImageConstPtr& labeled_msg,
                                                    const ImageLabelDistributionConstPtr & distribution_msg) {
    ros::Time curr_t = ros::Time::now();
    ROS_DEBUG_STREAM("Callback starts at time "<<uint32_t(curr_t.toSec())<<". "<<(uint32_t)curr_t.toNSec() );
    
    // std::cout << "depth: " << depth_msg->header.stamp << std::endl;
    // std::cout << "color: " << color_msg->header.stamp << std::endl;
    // std::cout << "camera_info: " << camera_info_msg->header.stamp << std::endl;
    // std::cout << "labeled_msg: " << labeled_msg->header.stamp << std::endl;
    // std::cout << "distribution_msg: " << distribution_msg->header.stamp << std::endl;


    // std::cout<<"DepthColorCallback: New callback"<< depth_msg->header.frame_id <<"\n";
    // Check for bad inputs
    if (depth_msg->header.frame_id != color_msg->header.frame_id) {
      ROS_ERROR("Depth iamge frame id [%s] doesn't match color image frame id [%s]",
          depth_msg->header.frame_id.c_str(), color_msg->header.frame_id.c_str());
      return;
    }
    
    // Get rgb and depth images
    cv_bridge::CvImagePtr color_ptr;
    cv_bridge::CvImagePtr depth_ptr;
    cv_bridge::CvImagePtr label_ptr;
    try{
      color_ptr = cv_bridge::toCvCopy(color_msg, sensor_msgs::image_encodings::RGB8);
      depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
      label_ptr = cv_bridge::toCvCopy(labeled_msg, sensor_msgs::image_encodings::TYPE_8UC1);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    //;cv::Mat color = color_ptr->image;
    //cv::Mat depth = depth_ptr->image;

    int rows = color_ptr->image.rows;
    int cols = color_ptr->image.cols;
    
    //float * buffer = new float [distribution_msg->distribution.data.layout.dims[0].stride];
    std::vector<float> data = distribution_msg->distribution.data;
   
    //  TensorMap<Tensor<int, 4>> t_4d(storage, 2, 4, 2, 8);
    //Eigen::Map<Eigen::MatrixXf> mat(data.data(), h, w);
    //cv::Mat distribution_output = cv::Mat(rows, cols, CV_32FC(uint32_t(distribution_msg->distribution.layout.dim[2].size)), distribution_msg->distribution.data).clone();
    cv::Mat distribution_output = cv::Mat(rows, cols, CV_32FC(NUM_CLASS), const_cast<float*>(distribution_msg->distribution.data.data())).clone();
    std::cout<<"distribution.data size "<<distribution_output.total() * distribution_output.elemSize()<<std::endl;
    
    //tf_infer->segmentation(color, 20, label_output, distribution_output);
    
      
    // Update camera model
    this->model_.fromCameraInfo(camera_info_msg);

    Depth2PointCloud1(depth_msg, color_msg, true, label_ptr->image, distribution_output);

    // Debugging
#ifndef NDEBUG
    Depth2PointCloud2(depth_msg, color_msg, false, label_ptr->image);
#endif
    //curr_t = ros::Time::now();
    //ROS_DEBUG_STREAM("Callback ends at time "<<uint32_t(curr_t.toSec())<<". "<<(uint32_t)curr_t.toNSec() );


  }

  

  template <unsigned int NUM_CLASS>
  inline void
  StereoSegmentation<NUM_CLASS>::Depth2PointCloud1(const sensor_msgs::ImageConstPtr& depth_msg,
                                                   const sensor_msgs::ImageConstPtr& color_msg,
                                                   bool publish_semantic,
                                                   const cv::Mat & label_img,
                                                   const cv::Mat & distribution_img) {
    std::cout<<"New image with label\n";
    
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

    sensor_msgs::ChannelFloat32 distribution_channel[NUM_CLASS];

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

    cv::Mat distribution_exp;
    if (publish_semantic){
      //cv::cv2eigen(distribution_img, distribution_exp);
      distribution_exp  = distribution_img;
      //cv::exp(distribution_img, distribution_exp);
      std::cout<<"elemSize: "<<distribution_exp.elemSize()<<", total "<<distribution_exp.total()<<", channels "<<distribution_exp.channels()<<std::endl;
    }

    // Iterate through depth image
    int i = 0;  // num of point
    int points_count = 0;
    for (int v = 0; v < int(depth_msg->height); ++v, depth_row += row_step, color += color_skip){
      for (int u = 0; u < int(depth_msg->width); ++u, color += color_step, ++i) {

      // Skip the upper half of the image
      //if (v < int(depth_msg->height / 2))
      //  continue;
      
      uint16_t depth = depth_row[u];

      // Check for invalid measurements
      // Kaiduo Fang: max_depth = 9;
      geometry_msgs::Point32 p;
      if (depth <= 0 ) {
        p.x = p.y = p.z = bad_point;
      }
      else {
        // Fill in XYZ
        p.x = (u - center_x) * depth * constant_x;
        p.y = (v - center_y) * depth * constant_y;
        p.z = (float) depth * unit_scaling;
        points_count += 1;
      }

      if (depth > 0)
      	if (sqrt(p.x * p.x + p.y * p.y + p.z * p.z) > 9)
          p.x = p.y = p.z = bad_point;

      cloud_msg->points.push_back(p);

      if (publish_semantic) {

        // Fill in semantics
        //cv::Mat dist_cv =distribution_exp(cv::Rect(u, v, 1, 1));
        cv::Vec<float, NUM_CLASS> dist_class = distribution_exp.at<cv::Vec<float, NUM_CLASS>>(v, u);
        /*        float sums = cv::sum(dist_class)[0];
        if (sums < 0.99 || sums > 1.01) {

          std::cerr<<"Wrong sum out of range: "<<sums<<std::endl;
          }*/
        //Eigen::VectorXf dist;
        //Eigen::Map<Eigen::Matrix<float, 20, 1> > eigenT( dist_cv.data );
        //dist = eigenT;
        //cv::cv2eigen(dist_cv, dist);
        //Eigen::VectorXf dist = distribution_exp(u, v);
        //Eigen::VectorXf dist_class = dist.segment(0, NUM_CLASS);
        /*
        Eigen::VectorXf dist_background = dist.segment(NUM_CLASS, dist.size()-NUM_CLASS);
        //assume background is 0
        float sum_background = dist_background.sum() + dist_class(0);
        dist_class(0) = sum_background;
        float sum_all_exp = dist_class.sum();
        dist_class = (dist_class / sum_all_exp).eval();
        */
        int max_ind =0;
        float max_elem = 0;
        for (int i = 0; i < NUM_CLASS; i++) {
          if (dist_class(i) > max_elem) {
            max_ind = i;
            max_elem = dist_class(i);
          }
          distribution_channel[i].values.push_back( dist_class(i) );
        }
        
        int label = max_ind;
        //std::cout<<"label is "<<label<<std::endl;
        label_channel.values.push_back(label);
        r_channel.values.push_back(std::get<0>(label2color[label]));
        g_channel.values.push_back(std::get<1>(label2color[label]));
        b_channel.values.push_back(std::get<2>(label2color[label]));

        
      } else {
        // Fill in r g b channels: bgr
        r_channel.values.push_back(color[0]);
        g_channel.values.push_back(color[1]);
        b_channel.values.push_back(color[2]);
        // Fill in semantics
        label_channel.values.push_back(1);
        distribution_channel[0].values.push_back(1);
        for (int i = 1; i < NUM_CLASS; i++) {
          distribution_channel[i].values.push_back(0);
        }
      }
    }
   }
    std::cout << "count_points: " << points_count << std::endl << std::endl;

    
    cloud_msg->channels.push_back(label_channel);
    cloud_msg->channels.push_back(r_channel);
    cloud_msg->channels.push_back(g_channel);
    cloud_msg->channels.push_back(b_channel);
    cloud_msg->channels.push_back(r_channel);
    cloud_msg->channels.push_back(g_channel);
    cloud_msg->channels.push_back(b_channel);
    for (int i = 0; i < NUM_CLASS; i++)
      cloud_msg->channels.push_back(distribution_channel[i]);

    
    pc1_publisher_.publish(cloud_msg);
  }

  
  
  template <unsigned int NUM_CLASS>
  inline void
  StereoSegmentation<NUM_CLASS>::Depth2PointCloud2(const sensor_msgs::ImageConstPtr& depth_msg,
                                                   const sensor_msgs::ImageConstPtr& color_msg,
                                                   bool publish_semantic,
                                                   const cv::Mat & label_img) {
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
   for (int v = 0 ; v < int(cloud_msg->height); ++v, depth_row += row_step, color += color_skip){
    for (int u = 0; u < int(cloud_msg->width); ++u, color += color_step, 
        ++iter_x, ++iter_y, ++iter_z, ++iter_a, ++iter_r, ++iter_g, ++iter_b) {
      
      // Skip the upper half of the image
      //if (v < int(cloud_msg->height / 2))
      // continue;

      uint16_t depth = depth_row[u];
      
      // Check for invalid measurements
      // update by Kaiduo Fang: depth > 9, max_depth;
      if (depth <= 0)
        *iter_x = *iter_y = *iter_z = bad_point;
      else {
        // Fill in XYZ
        *iter_x = (u - center_x) * depth * constant_x;
        *iter_y = (v - center_y) * depth * constant_y;
        *iter_z = (float) depth * unit_scaling;
        if (*iter_z > 10)
        *iter_z = bad_point;
      }
      

      // Fill in color
      //label_channel.values.push_back(label);
      if (publish_semantic) {
      uint8_t label = label_img.at<uint8_t>(u, v);

      *iter_a = 255;
      *iter_r = std::get<0>(label2color[label]);
      *iter_g = std::get<1>(label2color[label]);
      *iter_b = std::get<2>(label2color[label]);
      } else {
      *iter_a = 255;
      *iter_r = color[0];
      *iter_g = color[1];
      *iter_b = color[2];

      }
    
    }
   }
    pc2_publisher_.publish(cloud_msg);
  }





  
}


