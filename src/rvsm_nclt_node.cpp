/***********************************
 *
 * labeled_pc_map_node
 *
 ***********************************/
/* Author: Ray Zhang, rzh@umich.edu */



#include "labeled_pc_map.hpp"
#include "PointSegmentedDistribution.hpp"


#include <ros/console.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <sstream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <boost/shared_ptr.hpp>
#include <pcl/impl/point_types.hpp>

using namespace boost::filesystem;

// for nclt: 14 classes
POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::PointSegmentedDistribution<14>,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, rgb, rgb)
                                  (int, label, label)
                                  (float[14],label_distribution, label_distribution)
                                  )


void line2vector(std::string & input_line, std::vector<std::string>& tokens) {
  std::stringstream ss( input_line );

  while( ss.good() ) {
    std::string substr;
    std::getline( ss, substr, ',' );
    tokens.push_back( substr );
  }
}

template <unsigned int NUM_CLASS>
typename pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>>::Ptr
read_seg_pcd_text(const std::string & file_name) {
  // write to a pcd file
  typename pcl::PointCloud<pcl::PointSegmentedDistribution<14>>::Ptr in_cloud(new typename  pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>>);
  std::ifstream infile(file_name);
  typename pcl::PointSegmentedDistribution<NUM_CLASS> point;
  float rf;
  float gf;
  float bf;
  while(!infile.eof()) {
    infile >> point.x >> point.y >> point.z >> rf >> gf >> bf >> point.label;
    for (int j = 0 ;j < NUM_CLASS; j++) 
      infile >> point.label_distribution[j];
    uint8_t r = int(rf*255), g = int(gf*255), b = int(bf*255);
    // Example: Red color
    //std::cout << (int)r << " " <<(int) g << " " <<(int) b <<std::endl;
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    point.rgb = *reinterpret_cast<float*>(&rgb);
    in_cloud->push_back(point);
  }
  return in_cloud;
}


int main(int argc, char ** argv){

  ros::init(argc, argv, "nclt_map_node");
  ROS_INFO("nclt 14-class pc_painter init....");
  segmentation_projection::PointCloudPainter<14> painter;

  std::string seg_pcds(argv[1]);
  std::string pose_path(argv[2]);

  std::ifstream in_pose(pose_path);
  std::string pose_line;
  while (std::getline(in_pose, pose_line)) {

    std::stringstream ss(pose_line);    
    uint64_t pose_time;
    ss >> pose_time;
    std::string file =  seg_pcds + "/" + std::to_string(pose_time) + ".pcd" ;
    if ( !exists( file ) ) {
      std::cout << "Can't find pcd file "<<file<<" at time "<<pose_time << std::endl;
      continue;
    }

    //pcl::PointCloud<pcl::PointSegmentedDistribution<14>>::Ptr pc = read_seg_pcd_text<14>(file);
    pcl::PointCloud<pcl::PointSegmentedDistribution<14>>::Ptr pc(new pcl::PointCloud<pcl::PointSegmentedDistribution<14>>);
    pcl::io::loadPCDFile<pcl::PointSegmentedDistribution<14>> (file, *pc);    

    Eigen::Matrix4d T;
    T.setIdentity();
    for (int r = 0; r < 3; r++){
      for (int c = 0; c < 4; c++) {
        double pose;
        ss >> pose;
        T(r, c) = pose;
      }
    }
    Eigen::Affine3d aff(T.matrix());
    std::cout<<"Process "<<pc->size()<<" labeled points at time "<< pose_time<<std::endl;
    std::cout<<T<<std::endl;

                  
    painter.FuseMapIncremental(*pc, aff, ((double)pose_time) / 1e9);
    
  }

  //for (int i = 0; i!= 14; i++) {
  //  std::cout<<pc[0].label_distribution[i]<<std::endl;
  //}

  ros::spin();

  return 0;
}
