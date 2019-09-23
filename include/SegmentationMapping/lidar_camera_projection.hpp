#pragma once

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>
#include <Eigen/Dense>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/CameraInfo.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include "SegmentationMapping/ImageLabelDistribution.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <opencv2/core/matx.hpp>
#include <unordered_map>
#include <tuple>
#include <math.h>


using namespace std;
using namespace Eigen;
using namespace std::chrono;

namespace SegmentationMapping {
  
  class LidarProjection {
    public:
      LidarProjection()
      {
      	ros::NodeHandle nh("~");
        get_params(nh);
        
        // file.open("/home/fangkd/Desktop/time.txt");
        pcl_pub = nh.advertise<sensor_msgs::PointCloud>(Pub_Topic_, 5);
        pcl_sub.subscribe(nh, cloud_topic_, 50);
        img_sub.subscribe(nh, image_topic_, 50);
        cam_sub.subscribe(nh, cam_info_, 50);
        dist_sub.subscribe(nh, dist_info_, 5);
        sync_.reset(new Sync(MySyncPolicy(200), pcl_sub, img_sub, 
                             cam_sub, dist_sub));
        sync_->registerCallback(boost::bind(&LidarProjection::callback, 
                                            this, _1, _2, _3, _4));


    label2color[2]  =std::make_tuple(250, 250, 250 ); // road
    //label2color[3]  =std::make_tuple(128, 64,  128 ); // sidewalk
    label2color[3]  =std::make_tuple(250, 250,  250 ); // sidewalk
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

      void get_params(ros::NodeHandle &nh_);
      // Getting rostopics and extrinsic matrix from launch file.

      Matrix4f vector2matrix(vector<float> input);
      // Converting from [x y z qx qy qz qw] into transformation matrix.

      MatrixXf removePoints(const MatrixXf& matrix, vector<int> true_rows);

      MatrixXf dist_info(const MatrixXf& matrix, vector<int> true_rows);
      // vector true_rows contains the points idx we want to keep.
      // Discarding points from matrix.
      
      vector<int> depth_color(MatrixXf& depth);

      MatrixXf point_in_frame(const MatrixXf& point, const MatrixXf& dists);

      MatrixXf velo_point_filter(MatrixXf& velo_data);
      // Getting points in camera frame.
      
      void set_pointcloud_matrix(const sensor_msgs::PointCloud2ConstPtr& msg);

      void set_intrinsic(const sensor_msgs::CameraInfoConstPtr& cam);

      void visualize_results(const sensor_msgs::ImageConstPtr& img);

      void set_label_pcl(                    const sensor_msgs::ImageConstPtr& img,
                                             const ImageLabelDistribution::ConstPtr& info);

      void callback(const sensor_msgs::PointCloud2ConstPtr& msg, 
                    const sensor_msgs::ImageConstPtr& img,
                    const sensor_msgs::CameraInfoConstPtr& cam,
                    const ImageLabelDistribution::ConstPtr& info);

    private:
      int Lidar_max_range;

      string Pub_Topic_;
      string cloud_topic_;
      string image_topic_;
      string cam_info_;
      string dist_info_;

      Matrix4f Extrinsic;
      sensor_msgs::PointCloud Cloud;
      sensor_msgs::PointCloud DistriCloud;
      MatrixXf CloudMat;
      MatrixXf Intrinsic;
      vector<int> color_info;
      vector<int> pcl_info;
      std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> label2color;
  
      message_filters::Subscriber<sensor_msgs::PointCloud2> pcl_sub;
      message_filters::Subscriber<sensor_msgs::Image> img_sub;
      message_filters::Subscriber<sensor_msgs::CameraInfo> cam_sub;
      message_filters::Subscriber<sensor_msgs::Image> label_sub;
      message_filters::Subscriber<ImageLabelDistribution> dist_sub;

      typedef message_filters::sync_policies::ApproximateTime<
                                      sensor_msgs::PointCloud2,
                                      sensor_msgs::Image, 
                                      sensor_msgs::CameraInfo,
                                      ImageLabelDistribution> MySyncPolicy;
      typedef message_filters::Synchronizer<MySyncPolicy> Sync;
      boost::shared_ptr<Sync> sync_;
      ros::Publisher pcl_pub;

      const float MIN_DIS = 0;
      const float MAX_DIS = 120;
      int WIDTH, HEIGHT;
      // ofstream file;
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  };

  void LidarProjection::get_params(ros::NodeHandle &nh_){
    nh_.getParam("lidar_max_range", Lidar_max_range);
  	nh_.getParam("publish_to", Pub_Topic_);
    nh_.getParam("cloud_topic", cloud_topic_);
    nh_.getParam("image_topic", image_topic_);
    nh_.getParam("camera_info", cam_info_);
    nh_.getParam("distribution", dist_info_);
    nh_.getParam("image_width", WIDTH);
    nh_.getParam("image_height", HEIGHT);

    //vector<float> ImuLidar;
    //vector<float> ImuCamera;
    //nh_.getParam("imu2lidar", ImuLidar);
    //nh_.getParam("imu2camera", ImuCamera);
    vector<float> Extrinsic_param;
    nh_.getParam("extrinsic_mat", Extrinsic_param);
    
    // MatrixXf ImuLidarMat = vector2matrix(ImuLidar);
    // MatrixXf ImuCameraMat = vector2matrix(ImuCamera);
    //Extrinsic = ImuLidarMat * ImuCameraMat.inverse();
    Extrinsic.setIdentity();
    for (int i = 0; i < Extrinsic_param.size(); i++)
      Extrinsic(i / 4, i % 4) = Extrinsic_param[i];
    std::cout<<"Extrinsic calibration mat is \n"<<Extrinsic;
  }

  Matrix4f LidarProjection::vector2matrix(vector<float> input){
    // input is x y z qx qy qz qw;
    Matrix4f TransFull;
    Vector3f translation(input[0], input[1], input[2]);
    Quaternionf q;
    q.x() = input[3];
    q.y() = input[4];
    q.z() = input[5];
    q.w() = input[6];

    TransFull.setIdentity();
    TransFull.block<3, 3>(0, 0) = q.toRotationMatrix();
    TransFull.block<3, 1>(0, 3) = translation;

    return TransFull;
  }

  MatrixXf LidarProjection::removePoints(const MatrixXf& matrix, 
  	                                     vector<int> true_rows){
  	MatrixXf new_matrix;
    new_matrix.resize(2, true_rows.size());
    
    int flag_insert = 0;
    size_t flag_true = 0;
    for (size_t i = 0; i < matrix.cols(); i++){
      // cout << i << " ";
      if (i == true_rows[flag_true]){
        assert(flag_true <= matrix.cols());
        assert(flag_insert <= true_rows.size());
        if ((flag_insert >= true_rows.size())|| 
        	(flag_true >= new_matrix.cols())){
          break;
        }
        new_matrix(0, flag_insert) = matrix(0, i)/matrix(2, i);
        new_matrix(1, flag_insert) = matrix(1, i)/matrix(2, i);
        flag_insert += 1;
        flag_true += 1;
      }
      else{
        continue;
      }
    }    
    return new_matrix;
  }

  MatrixXf LidarProjection::dist_info(const MatrixXf& matrix, 
                                      vector<int> true_rows){
    MatrixXf new_matrix;
    new_matrix.resize(1, true_rows.size());

    int flag_insert = 0;
    size_t flag_true = 0;
    for (size_t i = 0; i < matrix.rows(); i++){
      
      if (i == true_rows[flag_true]){
        assert(flag_true <= matrix.rows());
        assert(flag_insert <= true_rows.size());
        if ((flag_insert >= true_rows.size()) ||
            (flag_true >= new_matrix.cols())){
          break;
        }
        new_matrix(0, flag_insert) = matrix(i, 0);
        flag_insert += 1;
        flag_true += 1;
      }
      else{
        continue;
      }
    }    
    return new_matrix;
  }

  vector<int> LidarProjection::depth_color(MatrixXf& depth){
    // cout << depth.rows() << " " << depth.cols() << endl;
    vector<int> color;
    for (int i = 0; i < depth.cols(); i++){
      if (depth(0, i) < MIN_DIS)
        color.push_back(MIN_DIS);
      else if (depth(0, i) > MAX_DIS)
        color.push_back(round(MAX_DIS * MAX_DIS/70));
      else
        color.push_back(round(depth(0, i) * MAX_DIS/70));
    }

    return color;
  }

  MatrixXf LidarProjection::point_in_frame(const MatrixXf& point, 
                                           const MatrixXf& dists){

    MatrixXf pointmat;
    pointmat.resize(4, point.cols());
    pointmat.block(0, 0, 3, point.cols()) = point;
    pointmat.block(3, 0, 1, point.cols()) = MatrixXf::Constant(
                                             1, point.cols(), 1);

    // pointmat.resize(point.rows(), 4);
    // pointmat.block(0, 0, point.rows(), 3) = point;
    // pointmat.block(0, 3, point.rows(), 1) = MatrixXf::Constant(
    //                                          point.rows(), 1, 1);
    // pointmat.transposeInPlace();
    pointmat = Intrinsic * Extrinsic * pointmat;
    // cout << "pointmat: " << pointmat.rows() << " " << pointmat.cols() << endl;
    vector<int> v;
    for (size_t i = 0; i < point.cols(); i++){
      if (pointmat(2, i) > 0){
        if (pointmat(0, i)/pointmat(2, i) >= 0
         && pointmat(0, i)/pointmat(2, i) < WIDTH-1
         && pointmat(1, i)/pointmat(2, i) >= 0
         && pointmat(1, i)/pointmat(2, i) < HEIGHT-1){
          v.push_back(i);          
        }
      }
    }
    // cout << "vector: " << v.size() << endl;
    MatrixXf inFrame = removePoints(pointmat, v);
    // cout << inFrame.rows() << " " << inFrame.cols() << endl;

    // MatrixXf new_dist = dist_info(dists, v);
    // cout << new_dist.rows() << " " << new_dist.cols() << endl;

    // color_info = depth_color(new_dist);
    pcl_info = v;
    // cout << inFrame.cols() << endl;
    return inFrame;
  }

  MatrixXf LidarProjection::velo_point_filter(MatrixXf& velo_data){
    // cout << velo_data.rows() << " " << velo_data.cols() << endl;
    MatrixXf x = velo_data.block(0, 0, 1, velo_data.cols());
    MatrixXf y = velo_data.block(1, 0, 1, velo_data.cols());
    MatrixXf z = velo_data.block(2, 0, 1, velo_data.cols());
  
    MatrixXf dist = (x.array().pow(2) + 
                     y.array().pow(2) + z.array().pow(2)).array().pow(0.5);
    // cout << dist.cols() << endl;
    // cout << velo_data.rows() << " " << velo_data.cols() << endl;
    MatrixXf points = point_in_frame(velo_data, dist);
  
    return points;
  }

  void LidarProjection::set_pointcloud_matrix(
  	   const sensor_msgs::PointCloud2ConstPtr& msg){
    sensor_msgs::convertPointCloud2ToPointCloud(*msg, Cloud);
    // CloudMat.resize(3, Cloud.points.size());

    vector<int> in_range;
    int count = 0;
    for (int i = 0; i < Cloud.points.size(); i++){
      if (sqrt(pow(Cloud.points[i].x, 2) + 
        pow(Cloud.points[i].y, 2) + pow(Cloud.points[i].z, 2)) <= Lidar_max_range){
        in_range.push_back(i);
        count += 1;
      }
    }
    CloudMat.resize(3, count);
    // for (int i = 0; i < Cloud.points.size(); i++){
    //   CloudMat(0, i) = Cloud.points[i].x;
    //   CloudMat(1, i) = Cloud.points[i].y;
    //   CloudMat(2, i) = Cloud.points[i].z;
    // }
    for (auto i = 0; i < in_range.size(); i++){
      CloudMat(0, i) = Cloud.points[in_range[i]].x;
      CloudMat(1, i) = Cloud.points[in_range[i]].y;
      CloudMat(2, i) = Cloud.points[in_range[i]].z;
    }
  }

  void LidarProjection::set_intrinsic(
  	   const sensor_msgs::CameraInfoConstPtr& cam){
  	Intrinsic.resize(12, 1);
    for (size_t j = 0; j < 12; j++){
      Intrinsic(j, 0) = cam->P[j];
    }
    Intrinsic.resize(4, 3);
    Intrinsic.transposeInPlace();
  }

  void LidarProjection::visualize_results(
  	   const sensor_msgs::ImageConstPtr& img){
  	cv::Mat image = cv_bridge::toCvCopy(img, "bgr8")->image;
    cv::cvtColor(image, image, cv::COLOR_BGR2HSV);
    for (size_t j = 0; j < color_info.size(); j++){
      cv::circle(image, cv::Point(round(CloudMat(0, j)),
                 round(CloudMat(1, j))), 2, 
                 cv::Scalar(color_info[j], 255, 255), -1);
    }
    cv::cvtColor(image, image, cv::COLOR_HSV2BGR);
    // cout << image.rows << " " << image.cols << endl; 
    cv::imshow("image", image);
    cv::waitKey(5);
  }

  void LidarProjection::set_label_pcl(
                                      const sensor_msgs::ImageConstPtr& img,
                                      const ImageLabelDistribution::ConstPtr& info)
  {
    int stride1 = info->distribution.layout.dim[1].stride;
    int stride2 = info->distribution.layout.dim[2].stride;
    int channel = info->distribution.layout.dim[2].size;
    DistriCloud.points.resize(CloudMat.cols());
    DistriCloud.channels.resize(channel+7);

    cv::Mat image = cv_bridge::toCvCopy(img, "rgb8")->image;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    
    for (int i = 0; i < channel+7; i++){
      DistriCloud.channels[i].values.resize(CloudMat.cols());
    }
    
    for (size_t j = 0; j < pcl_info.size(); j++){
      // std::cout<<j<<", \nnew "<<          info->distribution.data[4300799]<<std::endl;
      DistriCloud.points[j].x = Cloud.points[pcl_info[j]].x;
      DistriCloud.points[j].y = Cloud.points[pcl_info[j]].y;
      DistriCloud.points[j].z = Cloud.points[pcl_info[j]].z;
      int col = round(CloudMat(0, j));
      int row = round(CloudMat(1, j));
      float max_prob = 0;
      int max_l = -1;
      
      for (int k = 0; k < channel; k++){
        // std::cout<<"row "<<row<<", col "<<col<<", index "<<stride1 * row + stride2 * col + k<<std::endl;
        // std::cout<<info->distribution.data[stride1 * row + stride2 * col + k]<<"\n";
        DistriCloud.channels[k+7].values[j] = 
          info->distribution.data[stride1 * row + stride2 * col + k];

        if (info->distribution.data[stride1 * row + stride2 * col + k] > max_prob) {
          max_prob = info->distribution.data[stride1 * row + stride2 * col + k];

          max_l = k;
        }
      }
      // std::cout<<"max_l: "<<max_l<<std::endl;
      DistriCloud.channels[0].values[j] = max_l;
      DistriCloud.channels[1].values[j] = std::get<0>(label2color[max_l]);
      DistriCloud.channels[2].values[j] = std::get<1>(label2color[max_l]);
      DistriCloud.channels[3].values[j] = std::get<2>(label2color[max_l]);
      cv::Vec3b intensity = image.at<cv::Vec3b>(row, col);
      uchar blue = intensity.val[0];
      uchar green = intensity.val[1];
      uchar red = intensity.val[2];
      DistriCloud.channels[4].values[j] = red;
      DistriCloud.channels[5].values[j] = green;
      DistriCloud.channels[6].values[j] = blue;
      // std::cout<<"j"<<j<<std::endl;

      // debuging with visualization.
      // cv::circle(image, cv::Point(round(CloudMat(0, j)),
      //            round(CloudMat(1, j))), 2, 
      //            cv::Scalar(std::get<0>(label2color[max_l]), 
      //                       std::get<1>(label2color[max_l]), 
      //                       std::get<2>(label2color[max_l])), -1);
    }
    // std::cout<<"Finish generate distribution cloud\n";
    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    // cv::imshow("image", image);
    // cv::waitKey(5);
  }

  void LidarProjection::callback(const sensor_msgs::PointCloud2ConstPtr& msg,
                                 const sensor_msgs::ImageConstPtr& img,
                                 const sensor_msgs::CameraInfoConstPtr& cam,
                                 const ImageLabelDistribution::ConstPtr& info)
  {
    ROS_INFO("lidar callback!");
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    set_pointcloud_matrix(msg);

    set_intrinsic(cam);

    CloudMat = velo_point_filter(CloudMat);
    // cout << "CloudMat:" << CloudMat.rows() << " " << CloudMat.cols() << endl;
    // Visulization
    //visualize_results(img);

    set_label_pcl(img, info);
    DistriCloud.header = msg->header;
    DistriCloud.header.frame_id = "/velodyne_actual";
    pcl_pub.publish(DistriCloud);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>( t2 - t1 ).count();
    // cout << "1 Frame time: " << duration << endl;
    // cout << DistriCloud.header << endl;
    ROS_INFO("Published new cloud.");

    // file << duration << "  " << Cloud.points.size() << "  " << pcl_info.size() << endl;
  }

}
