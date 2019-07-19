#include "lidar_camera_projection.hpp"

int main(int argc, char** argv){
  ros::init(argc, argv, "lidar_projection");
  ROS_INFO("Start Projection");
  SegmentationMapping::LidarProjection Process;
  ros::spin();
  return 0;
}