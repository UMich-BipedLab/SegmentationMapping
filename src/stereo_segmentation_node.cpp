#include "stereo_segmentation.hpp"


int main (int argc, char** argv) {
  
  ros::init(argc, argv, "stereo_segmentation_node");
  ROS_INFO("Start stereo_seg node");
  SegmentationMapping::StereoSegmentation<14> stereo_seg;

  ros::spin();

  return 0;
}
