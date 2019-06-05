#pragma once

// std
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <ros/ros.h>
#include <ros/console.h>
#include <ros/time.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// boost
#include <boost/shared_ptr.hpp>


// tensorflow
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>



// for the PointSegmentedDistribution
#include "PointSegmentedDistribution.hpp"

namespace tf = tensorflow;
namespace tf_ops = tensorflow::ops;

namespace segmentation_projection {


  

  template <unsigned int NUM_CLASS>
  class StereoSegmentation {
  public:
    StereoSegmentation();
    ~StereoSegmentation();


  private:

    std::unique_ptr<tf::Session> tf_session;
    std::unique_ptr<tf::GraphDef> graph_def;
    std::string input_tensor_name;
    std::string output_tensor_name;
    std::vector<unsigned int> input_shape;
    void tf_init(const std::string & frozen_tf_graph,
                 const std::string & graph_intput_tensor,
                 const std::string & graph_output_tensor);
    cv::Ptr<cv::Mat> segmentation(const cv::Mat & rgb );
  
  };


  template <unsigned int NUM_CLASS>
  void tf_init(const std::string & frozen_tf_graph,
               const std::string & graph_intput_tensor,
               const std::string & graph_output_tensor){
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


    //declare
    mf there;
    m = &there;
    
    // fetch the input shape of the graph
    this->input_tensor_name = graph_intput_tensor;
    this->output_tensor_name = graph_output_tensor;
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
  cv::Ptr<cv::Mat> StereoSegmentation<NUM_CLASS>::segmentation(const cv::Mat & rgb) {
    // cv::mat assumes bgr!!

    // allocate a Tensor, without copy
    tf::Tensor input_img(tf::DT_FLOAT, tf::TensorShape({1,input_shape[1], input_shape[2] ,3}));
    float *raw_ptr = input_img.flat<float>().data();
    cv::Mat input_img_data(input_shape[1], input_shape[2], CV_32FC3, raw_ptr);
    rgb.convertTo(input_img_data, CV_32FC3);

    // neural net inference
    std::vector<tf::Tensor> outputs;
    TF_CHECK_OK(tf_session->Run({{ input_tensor_name, input_img }},
                                {output_tensor_name},
                                {},
                                &outputs));
    tf::Tensor & label_img = outputs[0];

    // convert to output cv mat
    cv::Ptr<cv::Mat> out_label_img(new cv::Mat);
    raw_ptr = label_img.flat<float>().data();
    cv::Mat out_label_original(label_img.dim_size(1), label_img.dim_size(2) , CV_32FC1, raw_ptr);
    out_label_original.copyTo(*out_label_img);
    
    
    /************ for debug use    **********************/
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", *out_label_img);                   // Show our image inside it.
    waitKey(2000);
    /****************************************************/
    

    return out_label_img;
    

  }


}
