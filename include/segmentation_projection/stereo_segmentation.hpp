#pragma once

// std
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cassert>
// ros
#include <ros/ros.h>
#include <ros/console.h>
#include <ros/time.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <cv_bridge/cv_bridge.h>

// opencv
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// boost
#include <boost/shared_ptr.hpp>

// tensorflow
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/cc/ops/standard_ops.h>

// for the PointSegmentedDistribution class
#include "PointSegmentedDistribution.hpp"


namespace tf = tensorflow;
namespace tf_ops = tensorflow::ops;

namespace segmentation_projection {

  typedef message_filters::sync_policies::ApproximateTime
  <sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image, sensor_msgs::CameraInfo> sync_pol;

  
  template <unsigned int NUM_CLASS>
  class StereoSegmentation {
  public:
    StereoSegmentation()
      : nh_() {
      ros::NodeHandle pnh("~");

      // init the message filter for image, depth, camera info
      color_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (pnh, "color_camera_image", 1);
      color_cam_sub_ = new message_filters::Subscriber<sensor_msgs::CameraInfo> (pnh, "color_camera_info", 1);
      depth_sub_ = new message_filters::Subscriber<sensor_msgs::Image> (pnh, "depth_color_image", 1);
      depth_cam_sub_ = new message_filters::Subscriber<sensor_msgs::CameraInfo> (pnh, "depth_camera_info", 1);
      sync_ = new message_filters::Synchronizer<sync_pol> (sync_pol(10), *color_sub_, *color_cam_sub_, *depth_sub_, *depth_cam_sub_);
      sync_->registerCallback(boost::bind(&StereoSegmentation::ColorDepthCallback, this,_1, _2, _3, _4));


      num_skip_frames = pnh.getParam("skip_every_k_frame", num_skip_frames);

      // init tensorflow 
      pnh.getParam("tf_input_tensor", input_tensor_name);
      pnh.getParam("tf_label_output_tensor", output_label_tensor_name);
      pnh.getParam("tf_distribution_output_tensor", output_distribution_tensor_name);
      std::string frozen_graph_path;
      pnh.getParam("tf_frozen_graph_path", frozen_graph_path );
      tf_init(frozen_graph_path);
      ROS_DEBUG_STREAM("Init tensorflow and revoke frozen graph from "<<frozen_graph_path);
    }
    
    ~StereoSegmentation() {
      delete sync_;
      delete color_sub_;
      delete color_cam_sub_;
      delete depth_sub_;
      delete depth_cam_sub_;
    }

    void ColorDepthCallback(const sensor_msgs::ImageConstPtr& color_msg,
                            const sensor_msgs::CameraInfoConstPtr& color_camera_info,
                            const sensor_msgs::ImageConstPtr& depth_msg,
                            const sensor_msgs::CameraInfoConstPtr& depth_camera_info);

    std::pair< std::shared_ptr<cv::Mat>, std::shared_ptr<Eigen::MatrixXf> > segmentation(const cv::Mat & rgb );
    
  private:
    ros::NodeHandle nh_;
    message_filters::Subscriber<sensor_msgs::Image>* color_sub_;
    message_filters::Subscriber<sensor_msgs::CameraInfo>* color_cam_sub_;
    message_filters::Subscriber<sensor_msgs::Image> *depth_sub_;
    message_filters::Subscriber<sensor_msgs::CameraInfo>* depth_cam_sub_;
    message_filters::Synchronizer<sync_pol>* sync_;

    int num_skip_frames;

    // for tensorflow
    std::unique_ptr<tf::Session> tf_session;
    std::unique_ptr<tf::GraphDef> graph_def;
    std::string input_tensor_name;
    std::string output_label_tensor_name;
    std::string output_distribution_tensor_name;
    std::vector<unsigned int> input_shape;
    void tf_init(const std::string & frozen_tf_graph);
  
  };

  template <unsigned int NUM_CLASS>
  void
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

    
  }


  template <unsigned int NUM_CLASS>
  void tf_init(const std::string & frozen_tf_graph) {
    // Initialize a tensorflow session
    tf::Session* session;
    tf::Status status = tf::NewSession(tf::SessionOptions(), &session);
    if (!status.ok()) {
      std::cerr << status.ToString() << "\n";
      return;
    }
    tf_session.reset(session);

    // read the trained frozen graph.pb file
    tf::GraphDef * graph_def_ptr = new tf::GraphDef;
    status = tf::ReadBinaryProto(tf::Env::Default(), frozen_tf_graph, graph_def_ptr);
    graph_def.reset(graph_def_ptr);
    if (!status.ok()) {
      std::cerr << status.ToString() << "\n";
      return ;
    }

    // Add the graph to the session
    status = tf_session->Create(*graph_def);
    if (!status.ok()) {
      std::cerr << status.ToString() << "\n";
      return;
    }


    // fetch the input shape of the graph
    for (int i = 0; i < graph_def->node_size(); i++) {
      std::cout<<graph_def->node(i).name()<<std::endl;
      if (graph_def->node(i).name() == input_tensor_name ) {
        auto shape = graph_def.node(i).Get(0).attr().at("shape").shape();
        std::cout<<"Find input tensor with shape "<<shape<<std::endl;
        input_shape.resize(4); // N, H, W, C
        input_shape[0] = shape.dim(0).size();
        input_shape[1] = shape.dim(1).size();
        input_shape[2] = shape.dim(2).size();
        input_shape[3] = shape.dim(3).size();
      }

    }
  }


  template <unsigned int NUM_CLASS>
  std::pair<
    std::shared_ptr<cv::Mat>,
    std::shared_ptr<Eigen::MatrixXf>
    >
  StereoSegmentation<NUM_CLASS>::segmentation(const cv::Mat & rgb) {
    // cv::mat assumes bgr!!
    assert (rgb.rows <= input_shape[1] || rgb.cols <= input_shape[2]);
    cv::Mat input_rgb = rgb;
    if (input_shape[1] > rgb.rows || input_shape[2] > rgb.cols) {
      int diff_width = - rgb.cols + input_shape[2];
      int diff_height = - rgb.rows + input_shape[1];
      input_rgb = cv::Mat(input_shape[1], input_shape[2], CV_32FC3, 0.0)
      cv::Mat pRoi = input_rgb(cv::Rect(diff_width / 2, diff_height / 2, rgb.cols, rgb.rows));
      pRoi = input_rgb.copy();
      cv::namedWindow("input to neural net", WINDOW_AUTOSIZE);
      cv::imshow("input_to_neural_net", input_rgb);
      cv::waitKey(500);
    }
    
    // allocate a Tensor, without copy
    tf::Tensor input_img(tf::DT_FLOAT, tf::TensorShape({1,input_shape[1], input_shape[2] ,3}));
    float *raw_ptr = input_img.flat<float>().data();
    cv::Mat input_img_data(input_shape[1], input_shape[2], CV_32FC3, raw_ptr);
    input_rgb.convertTo(input_img_data, CV_32FC3);

    // neural net inference
    std::vector<tf::Tensor> outputs;
    TF_CHECK_OK(tf_session->Run({{ input_tensor_name, input_img }},
                                {output_label_tensor_name, output_distribution_tensor_name},
                                {},
                                &outputs));
    tf::Tensor & label_img = outputs[0];
    tf::Tensor & distribution = outputs[1];

    // convert to output cv mat
    cv::Ptr<cv::Mat> out_label_img(new cv::Mat);
    raw_ptr = label_img.flat<float>().data();
    cv::Mat out_label_original(label_img.dim_size(1), label_img.dim_size(2) , CV_32FC1, raw_ptr);
    if (input_shape[1] > rgb.rows || input_shape[2] > rgb.cols) {
      int diff_width = - rgb.cols + input_shape[2];
      int diff_height = - rgb.rows + input_shape[1];
      //input_rgb = cv::Mat(input_shape[1], input_shape[2], CV_32FC3, 0.0)
      cv::Mat pRoi = out_label_original(cv::Rect(diff_width / 2, diff_height / 2, rgb.cols, rgb.rows));
      pRoi.copyTo(*out_label_img);
    }
    else
      out_label_original.copyTo(*out_label_img);
    
    
    /************ for debug use    **********************/
    cv::namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", *out_label_img);                   // Show our image inside it.
    cv::waitKey(2000);
    /****************************************************/
    

    return out_label_img;
    

  }





}
