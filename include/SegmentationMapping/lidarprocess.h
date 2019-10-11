#pragma once

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Dense>

#include <sensor_msgs/PointCloud.h>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include "SegmentationMapping/ImageLabelDistribution.h"
#include "PointSegmentedDistribution.hpp"

#include <chrono>
#include <iostream>
// #include <fstream>
#include <opencv2/core/matx.hpp>
#include <unordered_map>
#include <tuple>
#include <math.h>

namespace SegmentationMapping {
  
  class LidarProcess{
    
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
      LidarProcess()
      {
      	ros::NodeHandle nh("~");

      	set_params();
      	pcl_pub = nh.advertise<sensor_msgs::PointCloud>(Pub_Topic_, 5);
        
        label2color[2]  =std::make_tuple(250, 250, 250 ); // road
        //label2color[3]  =std::make_tuple(128, 64,  128 ); // sidewalk
        label2color[3]  =std::make_tuple(250, 250, 250 ); // sidewalk
        label2color[5]  =std::make_tuple(250, 128, 0   ); // building
        label2color[10] =std::make_tuple(192, 192, 192 ); // pole
        label2color[12] =std::make_tuple(250, 250, 0   ); // sign
        label2color[6]  =std::make_tuple(  0, 100, 0   ); // vegetation
        label2color[4]  =std::make_tuple(128, 128, 0   ); // terrain
        label2color[13] =std::make_tuple(135, 206, 235 ); // sky
        label2color[1]  =std::make_tuple( 30, 144, 250 ); // water
        label2color[8]  =std::make_tuple(220,  20, 60  ); // person
        label2color[7]  =std::make_tuple(  0,   0, 142 ); // car
        label2color[9]  =std::make_tuple(119,  11, 32  ); // bike
        label2color[11] =std::make_tuple(123, 104, 238 ); // stair
        label2color[0]  =std::make_tuple(255, 255, 255 ); // background
      }

      void set_params(); 
      // Set parameters by hard coding.

      // velo_point_filter: Intrinsic*Extrinsic*pcl_mat, then get rid
      // of the points out of frame.
      Eigen::MatrixXf velo_point_filter(Eigen::MatrixXf &velo_data);  

      // set_pcl_matrix: make Eigen::matrix to contain x, y, z in it.
      // used in Projection. 
      Eigen::MatrixXf set_pcl_matrix(typename pcl::PointCloud<pcl::PointXYZRGB> &pc_rgb);

      void set_label_pcl(const cv::Mat &img, const cv::Mat &dist_seg,
                         Eigen::MatrixXf &pcl_mat,
                         typename pcl::PointCloud<pcl::PointXYZRGB> &pc_rgb);

      void Projection(const cv::Mat &img, const cv::Mat &dist_seg, 
      	              typename pcl::PointCloud<pcl::PointWithXYZRGB> &pc_rgb);
      // image, distribution, pointcloud to do the projection.

    private:
      sensor_msgs::PointCloud DistriCloud;
      ros::Publisher pcl_pub;
      std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> label2color;
      std::vector<int> pcl_info;
      Eigen::MatrixXf Intrinsic;
      Eigen::Matrix4f Extrinsic;
      int WIDTH, HEIGHT;
  }

  void LidarProcess::set_params(){
  	Intrinsic.resize(3, 4);
  	Intrinsic << 616.3682, 0, 319.9346, 0,
  	             0, 616.7451, 243.6386, 0,
  	             0, 0, 1, 0;
  	std::vector<float> Extrinsic_param{
  		0.0388, -0.9992, -0.0075, 0.0738, 
  		0.1324, 0.0126, -0.9911, 0.0279, 
  		0.9904,0.0374,0.1328, -0.1829
  	};
  	Extrinsic.setIdentity();
  	for (int i = 0; i < Extrinsic_param.size(); i++){
  	  Extrinsic(i / 4, i % 4) = Extrinsic_param[i];
  	}
  	WIDTH = 640;
  	HEIGHT = 480;
  }

  void LidarProcess::Projection(const cv::Mat &img, const cv::Mat &dist_seg, 
  	                 typename pcl::PointCloud<pcl::PointXYZRGB> &pc_rgb)
  {
    // 1. convert pcl to Eigen
  	Eigen::MatrixXf pcl_mat = set_pcl_matrix(pc_rgb);

    // 2. remove points out of frame.
  	pcl_mat = velo_point_filter(pcl_mat);

    // 3. define new pointcloud msg to publish
  	set_label_pcl(img, dist_seg, pc_rgb, pcl_mat);


  	DistriCloud.header = pc_rgb.header;
  	DistriCloud.header.frame_id = "/velodyne_actual";
  	pcl_pub.publish(DistriCloud);
  }

  Eigen::MatrixXf LidarProcess::set_pcl_matrix(
  	typename pcl::PointCloud<pcl::PointXYZRGB> &pc_rgb){
  	Eigen::MatrixXf pcl_mat;
  	pcl.resize(4, pc_rgb.size());
  	for (int i = 0; i < pc_rgb.size(); i++){
  	  pcl_mat(0, i) = pc_rgb[i].x;
  	  pcl_mat(1, i) = pc_rgb[i].y;
  	  pcl_mat(2, i) = pc_rgb[i].z;
  	  pcl_mat(3, i) = 1;
  	}
  	return pcl_mat;
  }

  Eigen::MatrixXf velo_point_filter(Eigen::MatrixXf &pcl_mat){
  	// pcl_mat is (4, n).
  	pcl_mat = Intrinsic * Extrinsic * pcl_mat;
  	std::vector<int> v;
  	for (auto i = 0; i < pcl.mat.cols(); i++){
  	  if (pcl_mat(0, i) / pcl_mat(2, i) >= 0 &&
  	  	pcl_mat(0, i) / pcl_mat(2, i) < WIDTH - 1 &&
  	  	pcl_mat(1, i) / pcl_mat(2, i) >= 0 &&
  	  	pcl_mat(1, i) / pcl_mat(2, i) < HEIGHT - 1){
  	  	v.push_back(i);
  	  }
  	}
  	pcl_info = v;

  	// pcl_info: vector containing indx of points in the camera frame.
  	Eigen::MatrixXf in_frame;
    in_frame.resize(2, pcl_info.size());
    for (auto i = 0; i < pcl_info.size(); i++){
      int ind = pcl_info[i];
      if ( (ind >= pcl_mat.cols()) || (i >= pcl_info.size()) ){
        continue;
      }
      in_frame(0, i) = pcl_mat(0, ind)/pcl_mat(2, ind);
      in_frame(1, i) = pcl_mat(1, ind)/pcl_mat(2, ind);
    }
  	return in_frame;
  }

  void LidarProcess::set_label_pcl(const cv::Mat &img, 
                                   const cv::Mat &dist_seg, 
                                   Eigen::MatrixXf &pcl_mat,
         typename pcl::PointCloud<pcl::PointXYZRGB> &pc_rgb)
  {
  	cv::Mat image = img.clone();
    const int channel = dist_seg.channels();
  	DistriCloud.points.resize(pcl_mat.cols());
    DistriCloud.channels.resize(channel + 7);
    for (int i = 0; i < channel + 7; i++){
      DistriCloud.channels[i].values.resize(pcl_mat.cols());
    }

    int insert = 0;
    for (auto j = 0; j < pcl_info.size(); j++){
      int col = round(pcl_mat(0, j));
      int row = round(pcl_mat(1, j));
      float max_prob = 0;
      int max_l = -1;

      // somehow there will still be points outside frame. 
      if (row > 480 || col > 640 || row < 0 || col < 0)
        continue; 

      cv::Vec<float, channel> v = dist_seg.at<cv::Vec<float, channel> >(row, col);

      DistriCloud.points[insert].x = pc_rgb[pcl_info[j]].x;
      DistriCloud.points[insert].y = pc_rgb[pcl_info[j]].y;
      DistriCloud.points[insert].z = pc_rgb[pcl_info[j]].z;

      for (int k = 0; k < channel; k++){
        DistriCloud.channels[k + 7].values[j] = v[k];

        if (v[k] > max_prob){
          max_prob = v[k];
          max_l = k;
        }
      }
      DistriCloud.channels[0].values[insert] = max_l;
      DistriCloud.channels[1].values[insert] = std::get<0>(label2color[max_l]);
      DistriCloud.channels[2].values[insert] = std::get<1>(label2color[max_l]);
      DistriCloud.channels[3].values[insert] = std::get<2>(label2color[max_l]);
      cv::Vec3b intensity = image.at<cv::Vec3b>(row, col);
      uchar blue = intensity.val[0];
      uchar green = intensity.val[1];
      uchar red = intensity.val[2];
      DistriCloud.channels[4].values[insert] = red;
      DistriCloud.channels[5].values[insert] = green;
      DistriCloud.channels[6].values[insert] = blue;
      insert += 1;
      // debuging with visualization.
      // cv::circle(image, cv::Point(round(pcl_mat(0, j)),
      //            round(pcl_mat(1, j))), 2, 
      //            cv::Scalar(std::get<0>(label2color[max_l]), 
      //                       std::get<1>(label2color[max_l]), 
      //                       std::get<2>(label2color[max_l])), -1);
    }
    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    // cv::imshow("image", image);
    // cv::waitKey(5);
    // cout << "function run" << endl;
  }
}