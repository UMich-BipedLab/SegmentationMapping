#pragma once

// for the newly defined pointtype
#define PCL_NO_PRECOMPILE
#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <boost/shared_ptr.hpp>
#include <pcl/impl/point_types.hpp>




namespace pcl {
  template <unsigned int NUM_CLASS>
  struct PointSegmentedDistribution
  {
    PCL_ADD_POINT4D;                 
    PCL_ADD_RGB;
    int   label;
    float label_distribution[NUM_CLASS];   // templated on any number of classes. TODO
    //float label_distribution[14];
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
  } EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

}

namespace SegmentationMapping {
  
  template <unsigned int NUM_CLASS, typename PointWithXYZRGB>
  void PointSeg_to_PointXYZRGB(const pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>> & pc_seg,
                               typename pcl::PointCloud<PointWithXYZRGB> & pc_rgb) {
    pc_rgb.resize(pc_seg.size());
    for (int i = 0; i < pc_rgb.size(); i++) {
      auto & p_rgb = pc_rgb[i];
      auto & p_seg = pc_seg[i];

      p_rgb.x = p_seg.x;
      p_rgb.y = p_seg.y;
      p_rgb.z = p_seg.z;
      p_rgb.r = p_seg.r;
      p_rgb.g = p_seg.g;
      p_rgb.b = p_seg.b;
      
    }
    pc_rgb.header = pc_seg.header;
  }





}
