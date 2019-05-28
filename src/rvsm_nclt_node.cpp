/***********************************
 *
 * labeled_pc_map_node
 *
 ***********************************/
/* Author: Ray Zhang, rzh@umich.edu */



#include "labeled_pc_map.hpp"

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


int main(int argc, char ** argv){

  ros::init(argc, argv, "nclt_map_node");

  ROS_INFO("nclt 14-class pc_painter init....");
  segmentation_projection::PointCloudPainter<14> painter;

  std::string seg_pcds(argv[1]);
  std::string pose_path(argv[2]);

  std::ifstream in_pose(pose_path);
  
  std::vector<path> pcds;
  for(auto& entry : directory_iterator(seg_pcds) ) 
    pcds.push_back(entry.path());
  std::sort(pcds.begin(), pcds.end());
  for (int i = 0; i < pcds.size() ; i++) {

        
    std::string file =  pcds[i].string() ;
    
    std::size_t last_slash_pos = file.find_last_of("/");
    std::string time_sub_str = file.substr(last_slash_pos + 1, 19);
    std::cout<<"Frame "<<i<<" filename "<<file<<" at time "<<time_sub_str<<std::endl;
    std::stringstream sstime(time_sub_str);
    double time_max;
    sstime >> time_max;
    uint64_t actual_time = (uint64_t)(round(time_max / 1000) + 0.1);

    pcl::PointCloud<pcl::PointSegmentedDistribution<14>> pc;
    pcl::io::loadPCDFile<pcl::PointSegmentedDistribution<14>> (file, pc);

    std::string pose_line;
    std::getline(in_pose, pose_line);    
    std::stringstream ss(pose_line);    
    uint64_t pose_time;
    Eigen::Matrix4d T;
    T.setIdentity();
    ss >> pose_time;
    while ( pose_time != actual_time ) {
      std::cout<<"Pose time "<<pose_time << " does not match seg_pcd time "<<actual_time<<"\n";
      ss.clear();
      std::getline(in_pose, pose_line);
      if (in_pose.eof()) {
          std::cout<<"Ran out of pose file. exit...\n";
          return 0;
      }
      ss.str(pose_line);
      ss >> pose_time;
    }
    for (int r = 0; r < 3; r++){
      for (int c = 0; c < 4; c++) {
        double pose;
        char comma;
        ss >> comma;
        ss >> pose;
        T(r, c) = pose;
      }
    }
    Eigen::Affine3d aff(T.matrix());
    std::cout<<"Process "<<pc.size()<<" labeled points at time "<< pose_time<<", Pose is \n"<<T<<std::endl;
    painter.FuseMapIncremental(pc, aff, ((double)pose_time) / 1e6);
    
  }

  //for (int i = 0; i!= 14; i++) {
  //  std::cout<<pc[0].label_distribution[i]<<std::endl;
  //}

  ros::spin();

  return 0;
}
