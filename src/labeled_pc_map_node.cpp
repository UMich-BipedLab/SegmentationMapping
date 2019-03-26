/***********************************
 *
 * labeled_pc_map_node
 *
 ***********************************/
/* Author: Ray Zhang, rzh@umich.edu */



#include "labeled_pc_map.hpp"
#include <ros/console.h>
#include <iostream>

// for nclt: 14 classes
POINT_CLOUD_REGISTER_POINT_STRUCT(pcl::PointSegmentedDistribution<14>,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, rgb, rgb)
                                  (int, label, label)
                                  (float[14],label_distribution, label_distribution)
                                  )





int main(int argc, char ** argv){

  ros::init(argc, argv, "labeled_pc_map_node");

  ROS_INFO("nclt 14-class pc_painter init....");
  segmentation_projection::PointCloudPainter<14> painter;

  //pcl::PointCloud<pcl::PointSegmentedDistribution<14>> pc;
  //pcl::io::loadPCDFile ("/home/biped/.ros/segmented_pcd_cc/1335704128112920045.pcd", pc);
  //for (int i = 0; i!= 14; i++) {
  //  std::cout<<pc[0].label_distribution[i]<<std::endl;
  //}

  ros::spin();

  return 0;
}
