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
#include <ros/time.h>
#include <tf/transform_listener.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <std_msgs/Header.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/OccupancyGrid.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

#include <iostream>
#include <string>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <queue>
#include <tuple>
#include <boost/make_shared.hpp>

#include <eigen_conversions/eigen_msg.h>
#include  <tf_conversions/tf_eigen.h>
#include <tf/transform_datatypes.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

// for octmap
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap/ColorOcTree.h>
#include <octomap/SemanticOcTree.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_ros/conversions.h>

// for the PointSegmentedDistribution
#include "PointSegmentedDistribution.hpp"

// compute linear index for given map coords
#define MAP_IDX(sx, i, j) ((sx) * (j) + (i))

namespace SegmentationMapping {
  template<unsigned int NUM_CLASS>
  class PointCloudPainter {
  public:
    PointCloudPainter()
      : nh()
      , pnh("~")  
        //, painting_enabled_(true)
      , distribution_enabled_(false)
      , save_pcd_enabled_(false)
      , static_frame_("/map")
      , body_frame_("/body")
      , stacking_visualization_enabled_(true)
      , stacked_pc_ptr_(new pcl::PointCloud<pcl::PointXYZRGB>)
      , pointcloud_seg_stacked_ptr_(new typename pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>>)
      , path_visualization_enabled_(true)
      , color_octomap_enabled_(true)
      , occupancy_grid_enabled_(false)
      , cost_map_enabled_(false)
      , octomap_enabled_(false)
      , octomap_resolution_(0.1)
        //, octree_ptr_(new octomap::ColorOcTree(0.1))
      , octomap_frame_counter_(0)
      , octomap_num_frames_(200)
      , octomap_max_dist_(20.0)
    {
      // Parse parameters
      ros::NodeHandle pnh("~");
      //pnh.getParam("painting_enabled", painting_enabled_);
      pnh.getParam("static_frame", static_frame_);
      pnh.getParam("distribution_enabled", distribution_enabled_);
      pnh.getParam("body_frame", body_frame_);
      pnh.getParam("save_pcd_enabled", save_pcd_enabled_);
      pnh.getParam( "stacking_visualization_enabled", stacking_visualization_enabled_);
      pnh.getParam("path_visualization_enabled", path_visualization_enabled_);
      pnh.getParam("color_octomap_enabled", color_octomap_enabled_);
      pnh.getParam("occupancy_grid_enabled", occupancy_grid_enabled_);
      pnh.getParam("cost_map_enabled", cost_map_enabled_);
      pnh.getParam("octomap_enabled", octomap_enabled_);
      pnh.getParam("octomap_num_frames", octomap_num_frames_);
      pnh.getParam("octomap_max_dist", octomap_max_dist_);
      pnh.getParam("octomap_resolution", octomap_resolution_);

      this->pc_subscriber_ = pnh.subscribe("cloud_in", 100, &PointCloudPainter::PointCloudCallback, this);

      this->pc_publisher_ = pnh.advertise<sensor_msgs::PointCloud2>("cloud_out", 10);

      if (stacking_visualization_enabled_) {
        this->stacked_pc_publisher_ = pnh.advertise<sensor_msgs::PointCloud2>("stacked_pc_out", 10);
      }

      if (path_visualization_enabled_) {
        this->path_publisher_ = pnh.advertise<nav_msgs::Path>("path_out", 1);
      }

      if (color_octomap_enabled_) {
        color_octree_ptr_ = std::make_shared<octomap::ColorOcTree>(octomap::ColorOcTree(octomap_resolution_));
        color_octomap_publisher_ = pnh.advertise<octomap_msgs::Octomap>("color_octomap_out", 10);
        color_octree_ptr_->setOccupancyThres(0.52);
        double prob_hit = 0.5, prob_miss = 0.5;
        pnh.getParam("octomap_prob_hit", prob_hit);
        pnh.getParam("octomap_prob_miss", prob_miss);
        color_octree_ptr_->setProbHit(prob_hit);
        color_octree_ptr_->setProbMiss(prob_miss);
      }

      if (occupancy_grid_enabled_) {
        occupancy_grid_ptr_ = std::make_shared<nav_msgs::OccupancyGrid>(nav_msgs::OccupancyGrid());
        // set up occupancy grid info
        occupancy_grid_ptr_->info.resolution = 0.1;
        occupancy_grid_ptr_->info.width = 2000;
        occupancy_grid_ptr_->info.height = 2000;
        occupancy_grid_ptr_->info.origin.position.x = -100.0;
        occupancy_grid_ptr_->info.origin.position.y = -100.0;
        occupancy_grid_ptr_->info.origin.position.z = 0.0;
        occupancy_grid_ptr_->info.origin.orientation.x = 0.0;
        occupancy_grid_ptr_->info.origin.orientation.y = 0.0;
        occupancy_grid_ptr_->info.origin.orientation.z = 0.0;
        occupancy_grid_ptr_->info.origin.orientation.w = 1.0;
        occupancy_grid_ptr_->data.resize(occupancy_grid_ptr_->info.width * occupancy_grid_ptr_->info.height);
        std::fill(occupancy_grid_ptr_->data.begin(), occupancy_grid_ptr_->data.end(), -1);
        occupancy_grid_publisher_ = pnh.advertise<nav_msgs::OccupancyGrid>("occupancy_grid", 10);
      }


      if (cost_map_enabled_) {
        cost_map_ptr_ = std::make_shared<nav_msgs::OccupancyGrid>(nav_msgs::OccupancyGrid());
        // set up info
        cost_map_ptr_->info = occupancy_grid_ptr_->info;
        cost_map_ptr_->data.resize(cost_map_ptr_->info.width * cost_map_ptr_->info.height);
        std::fill(cost_map_ptr_->data.begin(), cost_map_ptr_->data.end(), 126);
        cost_map_publisher_ = pnh.advertise<nav_msgs::OccupancyGrid>("cost_map", 10);
      }


      if (octomap_enabled_) {
        // create label map
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


        octree_ptr_ = std::make_shared<octomap::SemanticOcTree>(octomap::SemanticOcTree(octomap_resolution_,
                                                                                        NUM_CLASS,
                                                                                        label2color));
        octomap_publisher_ = pnh.advertise<octomap_msgs::Octomap>("octomap_out", 10);
        octree_ptr_->setOccupancyThres(0.52);
        double prob_hit = 0.5, prob_miss = 0.5;
        pnh.getParam("octomap_prob_hit", prob_hit);
        pnh.getParam("octomap_prob_miss", prob_miss);
        octree_ptr_->setProbHit(prob_hit);
        octree_ptr_->setProbMiss(prob_miss);
        
      }
      
      ROS_INFO("ros_pc_map init finish\n");

    }

    void PointCloudCallback(const sensor_msgs::PointCloudConstPtr& cloud_msg);
    void FuseMapIncremental(const pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>> & pc, const Eigen::Affine3d & pose_at_pc, double stamp, bool is_write_centroids);
    const std::shared_ptr<octomap::SemanticOcTree> get_octree_ptr() const { return octree_ptr_; }

  
  private:
    ros::NodeHandle nh, pnh;
    ros::Subscriber pc_subscriber_;
    ros::Publisher pc_publisher_;
    tf::TransformListener listener_;

    std::string static_frame_;
    std::string body_frame_;
    
    //bool painting_enabled_;
    bool distribution_enabled_;
    bool save_pcd_enabled_;
    bool stacking_visualization_enabled_;
    bool path_visualization_enabled_;
    bool color_octomap_enabled_;
    bool occupancy_grid_enabled_;
    bool cost_map_enabled_;
    bool octomap_enabled_;

    // for voxel grid map
    ros::Publisher stacked_pc_publisher_;
    pcl::VoxelGrid<pcl::PointXYZRGB> voxel_;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr stacked_pc_ptr_;
    typename pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS> >::Ptr pointcloud_seg_stacked_ptr_;

    template <typename PointT>
    void merge_new_pc_to_voxel_grids(const typename pcl::PointCloud<PointT> & new_cloud, typename pcl::PointCloud<PointT> & stacked_cloud, const Eigen::Affine3d & T_map2body);

    // for path visualization
    ros::Publisher path_publisher_;
    nav_msgs::Path path_;
    void add_pose_to_path(const Eigen::Affine3d & T_map2body_new, const std_msgs::Header & header);

    // for color octomap
    std::shared_ptr<octomap::ColorOcTree> color_octree_ptr_;
    ros::Publisher color_octomap_publisher_;

    // for occupancy grid map
    std::shared_ptr<nav_msgs::OccupancyGrid> occupancy_grid_ptr_;
    ros::Publisher occupancy_grid_publisher_;
    std::shared_ptr<nav_msgs::OccupancyGrid> cost_map_ptr_;
    ros::Publisher cost_map_publisher_;
    std::queue<int> grids_queue_;
    std::vector<int> find_neighbors(int index, int width, int height);
    void update_cost_map();


    // for semantic octomap
    std::shared_ptr<octomap::SemanticOcTree> octree_ptr_;
    //std::shared_ptr<octomap::ColorOcTree> octree_ptr_;
    int octomap_frame_counter_;
    int octomap_num_frames_;
    double octomap_resolution_;
    ros::Publisher octomap_publisher_;
    float octomap_max_dist_;
    std::unordered_map<int, std::tuple<uint8_t, uint8_t, uint8_t>> label2color;
    void add_pc_to_octomap(const pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>> & pc_rgb,
                           const Eigen::Affine3d & T_eigen ,
                           bool is_update_occupancy,
                           bool is_write_centroids,
                           const ros::Time & stamp);
  };
  


  template <unsigned int NUM_CLASS> template<typename PointT>
  inline void
  PointCloudPainter<NUM_CLASS>::merge_new_pc_to_voxel_grids(const typename pcl::PointCloud<PointT> & new_cloud, typename pcl::PointCloud<PointT> & stacked_pc, const Eigen::Affine3d & T_eigen){
    typename pcl::PointCloud<PointT> transformed_cloud;
    pcl::transformPointCloud (new_cloud, transformed_cloud , T_eigen);
    stacked_pc = stacked_pc + transformed_cloud;

    // Voxel Grid filtering: uncommet this if the input is sparse
    //pcl::PCLPointCloud2::Ptr cloud (new pcl::PCLPointCloud2 ());
    //pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());
    //pcl::toPCLPointCloud2(stacked_pc, *cloud);    
    // Create the filtering object
    //pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    //sor.setInputCloud (cloud);
    //sor.setLeafSize (0.1f, 0.1f, 0.1f);
    //sor.filter (*cloud_filtered);
    //pcl::fromPCLPointCloud2(*cloud_filtered, stacked_pc);
    
  }

  template <unsigned int NUM_CLASS>
  inline void
  PointCloudPainter<NUM_CLASS>::add_pose_to_path(const Eigen::Affine3d  & T_eigen, const std_msgs::Header & header) {
 
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = header.stamp;
    pose_stamped.header.frame_id = this->body_frame_;
    tf::poseEigenToMsg (T_eigen, pose_stamped.pose);
    //std::cout<<"xyz: "<<T_eigen(0, 3)<<", "<<T_eigen(1, 3)<<", "<<T_eigen(2, 3)<<", at time:"<< header.stamp.toNSec()<<" \n";

    path_.header.stamp = header.stamp;
    path_.header.frame_id = this->static_frame_;
    path_.poses.push_back(pose_stamped);

    path_publisher_.publish(path_);
      
  }

  template<unsigned int NUM_CLASS>
  inline std::vector<int>
  PointCloudPainter<NUM_CLASS>::find_neighbors(int index, int width, int height) {
    std::vector<int> neighbors{index-width-1, index-width, index-width+1,
                               index-1, index+1,
                               index+width-1, index+width, index+width+1};
    for (auto it = neighbors.begin(); it != neighbors.end(); ) {
      if ((*it >= 0) && ( *it < width * height))
        ++it;
      else
        it = neighbors.erase(it);
    }
    return neighbors;
  }


  template<unsigned int NUM_CLASS>
  inline void
  PointCloudPainter<NUM_CLASS>::update_cost_map() {
    std::queue<int> grids_queue;
    std::vector<float> distance;
    distance.resize(cost_map_ptr_->info.width * cost_map_ptr_->info.height);

    // Initialize distance
    for (int i = 0; i < occupancy_grid_ptr_->info.width * occupancy_grid_ptr_->info.height; ++i) {
      if (occupancy_grid_ptr_->data[i] > 0)  // occupied
        distance[i] = 0.0f;
      else
        distance[i] = std::numeric_limits<float>::infinity();
    }
    
    for (int i = 0; i < occupancy_grid_ptr_->info.width * occupancy_grid_ptr_->info.height; ++i) {
      if (distance[i] == 0) {
        std::vector<int> neighbors = find_neighbors(i, occupancy_grid_ptr_->info.width, occupancy_grid_ptr_->info.height);
        for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
          if (distance[*it] == std::numeric_limits<float>::infinity())
            grids_queue.push(*it);
        }
      }
    }

    while(!grids_queue.empty()) {
      int grid = grids_queue.front();
      grids_queue.pop();
      if (distance[grid] == std::numeric_limits<float>::infinity()) {
        std::vector<int> neighbors = find_neighbors(grid, occupancy_grid_ptr_->info.width, occupancy_grid_ptr_->info.height);
        float min = std::numeric_limits<float>::infinity();
        bool found_min = false;
        for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
          if (distance[*it] < min) {
            min = distance[*it];
            found_min = true;
          }
          if (distance[*it] == std::numeric_limits<float>::infinity())
            grids_queue.push(*it);
        }

        //std::cout << "queue size: " << grids_queue.size() << std::endl;
        if (found_min)
          distance[grid] = 1 + min;
      }
    }
    
    // Write distance to cost map
    for (int i = 0; i < cost_map_ptr_->info.width * cost_map_ptr_->info.height; ++i) {
      if (distance[i] <= 126)
        cost_map_ptr_->data[i] = (int8_t) distance[i];
      else
        cost_map_ptr_->data[i] = 126;
    }
  }

  

  template<unsigned int NUM_CLASS>
  inline void
  PointCloudPainter<NUM_CLASS>::add_pc_to_octomap(const pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>> & pc_rgb,
                                                  const Eigen::Affine3d & T_map2body_eigen,
                                                  bool is_update_occupancy,
                                                  bool is_write_centroids,
                                                  const ros::Time & stamp) {

    pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>> transformed_pc;
    pcl::transformPointCloud (pc_rgb, transformed_pc, T_map2body_eigen);

    std::cout<<"Add "<<pc_rgb.size()<<" points to octomap \n";
    for (int i = 0; i < transformed_pc.size(); i++  ) {
      pcl::PointSegmentedDistribution<NUM_CLASS> p = transformed_pc[i];

      uint32_t rgb = *reinterpret_cast<int*>(&p.rgb);
      uint8_t r = (rgb >> 16) & 0x0000ff;
      uint8_t g = (rgb >> 8)  & 0x0000ff;
      uint8_t b = (rgb)       & 0x0000ff;
      float x = p.x;
      float y = p.y;
      float z = p.z; // for NCLT only
      octomap::point3d endpoint ( x,  y, z);

      if (color_octomap_enabled_) {
        octomap::ColorOcTreeNode* n = color_octree_ptr_->updateNode(endpoint, true);
        color_octree_ptr_->averageNodeColor(x, y, z, r, g, b);
      }

      if (occupancy_grid_enabled_) {
        int map_index = MAP_IDX(occupancy_grid_ptr_->info.width,
                                int( (x-occupancy_grid_ptr_->info.origin.position.x) /  occupancy_grid_ptr_->info.resolution ),
                                int( (y-occupancy_grid_ptr_->info.origin.position.y) /  occupancy_grid_ptr_->info.resolution)) ;

        if (map_index < occupancy_grid_ptr_->info.width * occupancy_grid_ptr_->info.height) {
          //std::cout << "x: " << x << std::endl;
          //std::cout << "y: " << y << std::endl;
          //std::cout << "map index: " << map_index;
          if (z < 2.0 && (p.label == 3 || p.label == 2) )
            occupancy_grid_ptr_->data[map_index] = 0;
          else {
            occupancy_grid_ptr_->data[map_index] = 100;
          }
        }
      }

      if (octomap_enabled_) {
        if (is_update_occupancy)
          octree_ptr_->updateNode(endpoint, true); // integrate 'occupied' measurement
        
        octomap::SemanticOcTreeNode * result = octree_ptr_->search(endpoint);
        std::vector<float> label_dist(p.label_distribution, std::end(p.label_distribution));
        if (i == 0) {
          std::cout<<"Before recurrent tree update, @ "<<x<<", "<<y << ", "<<z<<" the distribution is  ";
          std::cout<< result->getSemantics()<<"\n";
        }
        //octree_ptr_->averageNodeColor(result, r, g, b);
        octree_ptr_->averageNodeSemantics(result, label_dist );

        //octree_ptr_->averageNodeColor( x,  y,  z, r, g, b);
      }
    }

    if (color_octomap_enabled_) {
      std::cout<<"publishing color octree\n";
      octomap_msgs::Octomap cmap_msg;
      cmap_msg.binary = 0 ;
      cmap_msg.resolution = 0.1;
      octomap_msgs::fullMapToMsg(*color_octree_ptr_, cmap_msg);
      cmap_msg.header.frame_id = this->static_frame_;
      cmap_msg.header.stamp = stamp;
      color_octomap_publisher_.publish(cmap_msg);
    }

    if (occupancy_grid_enabled_) {
      std::cout << "publishing occupancy grid\n";
      occupancy_grid_ptr_->header.frame_id = this->static_frame_;
      occupancy_grid_ptr_->header.stamp = stamp;
      occupancy_grid_publisher_.publish(*occupancy_grid_ptr_); 
    }

    if (cost_map_enabled_) {
      update_cost_map();
      std::cout << "publishing cost map\n";
      cost_map_ptr_->header.frame_id = this->static_frame_;
      cost_map_ptr_->header.stamp = stamp;
      cost_map_publisher_.publish(*cost_map_ptr_); 
    }

    if (octomap_enabled_){
      if (is_update_occupancy)
        octree_ptr_->updateInnerOccupancy();
      
      
      if (is_write_centroids) {
        uint64_t get_usec = (uint64_t)stamp.toNSec() / 1000;
        std::ofstream centroids_file("octree_centroids/" + std::to_string(get_usec) + ".txt");
        
        for (octomap::SemanticOcTree::leaf_iterator it = octree_ptr_->begin_leafs(),
               //for (octomap::ColorOcTree::leaf_iterator it = octree_ptr_->begin_leafs(),
               end=octree_ptr_->end_leafs(); it!= end; ++it)  {
          centroids_file << it.getX() <<" "
                         << it.getY() <<" "
                         << it.getZ() <<"\n";

        }
        centroids_file.close();
                                         
      }

      // for debugging
      for (int i = 0; i < 1; i++  ) {
        pcl::PointSegmentedDistribution<NUM_CLASS> p = transformed_pc[i];

        float x = p.x;
        float y = p.y;
        float z = p.z; // for NCLT only
        octomap::point3d endpoint ( x,  y, z);

        octomap::SemanticOcTreeNode * result = octree_ptr_->search(endpoint);

        std::cout<<"After recurrent tree update, @ "<<x<<", "<<y << ", "<<z<<" the distribution is  ";
        for (auto && d : result->getSemantics().label) {
          std::cout<<d<<" ";
        }
        std::cout<<"\n";
      }

      std::cout<<"publishing octree\n";
      octomap_msgs::Octomap bmap_msg;
      bmap_msg.binary = 0 ;
      bmap_msg.resolution = 0.1;
      octomap_msgs::fullMapToMsg(*octree_ptr_, bmap_msg);
      bmap_msg.header.frame_id = this->static_frame_;
      bmap_msg.header.stamp = stamp;
      octomap_publisher_.publish(bmap_msg);
    }
  }

  template <unsigned int NUM_CLASS>
  inline void
  PointCloudPainter<NUM_CLASS>::FuseMapIncremental(const pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>> & pc,
                                                   const Eigen::Affine3d & pose_at_pc, double stamp,
                                                   bool is_write_centroids){
    
    ros::Time t(stamp);
    if (path_visualization_enabled_) {
      std_msgs::Header header;
      header.stamp = t;
      header.frame_id = this->body_frame_;
      add_pose_to_path(pose_at_pc, header);
    }

    if (octomap_enabled_ && distribution_enabled_) {
      add_pc_to_octomap(pc , pose_at_pc, true, is_write_centroids, t );
      this->octomap_frame_counter_++;
      //if (this->octomap_frame_counter_ > this->octomap_num_frames_) {
        octree_ptr_->write("semantic_octree.ot");
        //}

    }

    if (stacking_visualization_enabled_ && distribution_enabled_) {
      std_msgs::Header header;
      header.stamp = t;
      header.frame_id = this->static_frame_;
      pcl_conversions::toPCL(header, pointcloud_seg_stacked_ptr_->header);
      pointcloud_seg_stacked_ptr_->header.frame_id = this->static_frame_;
      merge_new_pc_to_voxel_grids<pcl::PointSegmentedDistribution<NUM_CLASS>>(pc,
                                                                              *pointcloud_seg_stacked_ptr_,
                                                                              pose_at_pc);
      pcl::io::savePCDFile ("segmented_pcd_stacked/stacked_pc_distribution.pcd", *pointcloud_seg_stacked_ptr_);

    }

    // publish pointxyzrgb raw at the current timestamp, for debugging visualization
    sensor_msgs::PointCloud2 painted_cloud;
    pcl::PointCloud<pcl::PointXYZRGB> pc_rgb;
    PointSeg_to_PointXYZRGB<NUM_CLASS, pcl::PointXYZRGB>(pc, pc_rgb);
    pcl::toROSMsg(pc_rgb, painted_cloud);
    painted_cloud.header.frame_id = this->body_frame_;
    pc_publisher_.publish(painted_cloud);

    
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
      this->listener_.lookupTransform(this->static_frame_, this->body_frame_,  
                                      cloud_msg->header.stamp, transform);
    }
    catch (tf::TransformException ex){
      std::cout<<"tf look for failed\n";
      ROS_ERROR("%s",ex.what());
      return;
    }
    Eigen::Affine3d T_map2body_eigen;
    tf::transformTFToEigen (transform,T_map2body_eigen);
  
    //for (int j = 0; j < img.rows; ++j) {
    std::cout<<"At time "<<cloud_msg->header.stamp.toSec()<<", # of lidar pts is "<<cloud_msg->points.size()<<std::endl;
    for (int i = 0; i < cloud_msg->points.size(); ++i) {
      pcl::PointXYZRGB p;
      p.x = cloud_msg->points[i].x;
      p.y = cloud_msg->points[i].y;
      p.z = cloud_msg->points[i].z;

      // filter out points that are too far away
      if (p.x * p.x + p.y * p.y + p.z * p.z > octomap_max_dist_ * octomap_max_dist_ )
        continue;

      // filter out sky, background, human
      int label = cloud_msg->channels[0].values[i];
      if (label == 13 || label== 0 || label == 8 )
        continue;

      
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
          //std::cout<<p_seg.label_distribution[c]<<std::endl;
        }
        //if (sums < 0.99 || sums > 1.01) {
          //std::cout<<sums<<std::endl;
        //  continue;
        //}
        
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
      pcl_conversions::toPCL(cloud_msg->header, pointcloud_pcl.header);
      pointcloud_pcl.header.frame_id = this->static_frame_;
      this->merge_new_pc_to_voxel_grids<pcl::PointXYZRGB>(pointcloud_pcl, *stacked_pc_ptr_, T_map2body_eigen);            
      pcl::toROSMsg(*(this->stacked_pc_ptr_), stacked_cloud);
      stacked_cloud.header = cloud_msg->header;
      stacked_cloud.header.frame_id = this->static_frame_;
      stacked_pc_publisher_.publish((stacked_cloud));
     
      if (distribution_enabled_ && save_pcd_enabled_) {
        pcl_conversions::toPCL(cloud_msg->header, pointcloud_seg_stacked_ptr_->header);
        pointcloud_seg_stacked_ptr_->header.frame_id = this->static_frame_;
        merge_new_pc_to_voxel_grids<pcl::PointSegmentedDistribution<NUM_CLASS>>(pointcloud_seg,
                                                                                *pointcloud_seg_stacked_ptr_,
                                                                                T_map2body_eigen);

        ROS_INFO("Save PCD file for the stacked pointcloud with label distribution");
        pcl::io::savePCDFile ("segmented_pcd/stacked_pc_distribution.pcd", *pointcloud_seg_stacked_ptr_);
        pcl::io::savePCDFile ("segmented_pcd/stacked_pc_rgb.pcd", pointcloud_pcl);
          

      }

    }

    // publish the path
    if (this->path_visualization_enabled_) 
      this->add_pose_to_path(T_map2body_eigen,  cloud_msg->header);

    
    // save the result of each time step as pcd files
    if (this->save_pcd_enabled_) {
      std::string name_pcd = std::to_string(cloud_msg->header.stamp.toNSec());
      pcl::io::savePCDFile ("segmented_pcd/" + name_pcd + ".pcd", pointcloud_seg);
    }

    // produce the color octomap. 
    if (this->octomap_enabled_ || this->color_octomap_enabled_) {
      this->add_pc_to_octomap(pointcloud_seg,T_map2body_eigen, true, false, cloud_msg->header.stamp);
      this->octomap_frame_counter_++;
      //if (this->octomap_frame_counter_ == this->octomap_num_frames_) {
        //octree_ptr_->write("semantic_octree.ot");
      //}
    }

  }


} 


