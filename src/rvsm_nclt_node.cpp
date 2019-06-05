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
using namespace octomap;
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

void PrintQueryInfo(point3d query, SemanticOcTreeNode* node) {
  if (node != NULL) {
    std::cout << "occupancy probability at " << query << ":\t " << node->getOccupancy() << std::endl;
    std::cout << "color of node is: " << node->getColor() << std::endl;    
    std::cout << "semantics of node is: " << node->getSemantics() << std::endl;
  }
  else 
    std::cout << "occupancy probability at " << query << ":\t is unknown" << std::endl;
}
  


void QueryPointCloudSemantics(const std::shared_ptr< SemanticOcTree> tree,
                              const pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                              const pcl::PointCloud<pcl::PointXYZ>::Ptr pc_local,
                              std::string file_name) {
  
  std::string txt_file = file_name.substr(0, file_name.size()-8) + ".txt";
  std::cout << txt_file << std::endl;
  std::ofstream data_out(txt_file);

  std::cout << "Performing some queries ... # of points is "<<pc->size()<<", writing to " << txt_file<<std::endl;
  int count = 0;
  for (int j = 0; j < pc->points.size(); ++j) {
    point3d query (pc->points[j].x, pc->points[j].y, pc->points[j].z);
    const SemanticOcTreeNode* node = tree->search(query);
    if (node != NULL && node->getOccupancy() > 0.35 && node->isSemanticsSet()) {
      count += 1;
      //PrintQueryInfo(query, node);
      auto semantics = node->getSemantics();
      data_out << pc_local->points[j].x << " "
               << pc_local->points[j].y << " "
               << pc_local->points[j].z;
          
      for (int c = 0; c < semantics.label.size(); ++c)
        data_out << " " << semantics.label[c];
      data_out<<"\n";
    } else
      continue;
  }
  std::cout<<"Count is "<<count<<std::endl;
  data_out.close();
}


std::unordered_map<uint64_t, std::vector<double>> ReadPointCloudPoses(std::string pose_file) {
  std::ifstream pose_in(pose_file);
  std::string pose_line;
  std::unordered_map<uint64_t, std::vector<double>> time2pose;
  
  while (std::getline(pose_in, pose_line)) {
    std::stringstream ss(pose_line);
    uint64_t t;
    ss >> t;
    uint64_t time = t;  // (uint64_t) (std::round((double)t / 1000.0) + 0.1);
    
    std::vector<double> pose;
    for (int i = 0; i < 12; ++i) {
      double ele;
      ss >> ele;
      pose.push_back(ele);
    }
    time2pose[time] = pose;
  }
  return time2pose;
}


pcl::PointCloud<pcl::PointXYZ>::Ptr read_nclt_pc_file(const std::string & filename) {
  std::ifstream infile(filename);

  pcl::PointCloud<pcl::PointXYZ>::Ptr pc_ptr(new pcl::PointCloud<pcl::PointXYZ>);
  pc_ptr->header.frame_id = "velodyne";
  //std::stringstream ss(filename.substr(0, filename.size()-4));
  //pc_ptr->header.stamp << ss;
  //std::cout<<"read lidar points at time stamp "<<pc_ptr->header.stamp<<". ";

  std::string line;

  while (std::getline(infile, line)) {
    std::stringstream ss(line);
    pcl::PointXYZ p;
    float x, y, z;
    char comma;
    if (!(ss >> x) ||
        !(ss >> y) ||
        !(ss >> z)) {
      std::cerr<<"sstream convert "<<line<<" to floats fails\n";
      return pc_ptr;
    }
    float intensity, l;
    ss >> intensity >> l;
    p.x = x;
    p.y = y;
    p.z = z;
    pc_ptr->push_back(p);

  }
  //std::cout<<" # of points is "<<pc_ptr->size()<<std::endl;

  infile.close();
  return pc_ptr;
}



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
  typename pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>>::Ptr in_cloud(new typename  pcl::PointCloud<pcl::PointSegmentedDistribution<NUM_CLASS>>);
  std::ifstream infile(file_name);
  typename pcl::PointSegmentedDistribution<NUM_CLASS> point;
  float rf;
  float gf;
  float bf;
  while(!infile.eof()) {

    infile >> point.x >> point.y >> point.z >> rf >> gf >> bf >> point.label;
    //infile >> point.x >> point.y >> point.z >> point.label;
    for (int j = 0 ;j < NUM_CLASS; j++) 
      infile >> point.label_distribution[j];
    uint8_t r = int(rf*255), g = int(gf*255), b = int(bf*255);
    //uint8_t r = int(250), g = int(250), b = int(250);
    // Example: Red color
    //std::cout << (int)r << " " <<(int) g << " " <<(int) b <<std::endl;
    uint32_t rgb = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    point.rgb = *reinterpret_cast<float*>(&rgb);
    in_cloud->push_back(point);
    //std::cout<<"file "<<file_name<<", in_cloud->size is "<<in_cloud->size()<<", xyz "<<point.x<<","<<point.y<<","<<point.z<<std::endl;
  }
  return in_cloud;
}


void query_test(const std::shared_ptr<octomap::SemanticOcTree> octree_ptr ) {
  
  // Read pc poses
  std::string pose_file = "/home/biped/perl_code/workspace/src/segmentation_projection/data/nclt_04_29/pose_sequence_16.txt";
  std::unordered_map<uint64_t, std::vector<double>> time2pose = ReadPointCloudPoses(pose_file);
  std::cout << "Query: Read poses size: " << time2pose.size() << std::endl;

  // Read gt times
  std::string time_file = "/home/biped/perl_code/rvsm/nclt_may23_gt/gt_times.txt";
  std::ifstream time_in(time_file);
  std::string time_line;
  //std::string seq_path("/home/biped//perl_code/gicp/visualization/data/rvm_04_29/");
  std::string seq_path("/home/biped//perl_code/workspace/src/segmentation_projection/data/nclt_04_29/nclt_lidar_downsampled/");
  
  while (std::getline(time_in, time_line)) {
    std::stringstream ss(time_line);
    uint64_t time;
    ss >> time;
    std::string csv_file = seq_path + "/" + std::to_string(time) + ".bin.csv";
    if (!exists(csv_file)) {
      std::cout << "Can't find csv file " << csv_file << " at time " << time << std::endl;
      continue;
    } 

    pcl::PointCloud<pcl::PointXYZ>::Ptr pc = read_nclt_pc_file(csv_file);
    //std::cout << "Query:: Process " << pc->size() << " points at time " << time << std::endl;
    Eigen::Matrix4d T;
    T.setIdentity();
    for (int r=0; r<3; ++r){
      for (int c=0; c<4; ++c) {
        T(r, c) = (time2pose.at(time))[4*r+c];
      }
    }
    Eigen::Affine3d aff(T.matrix());
    
    
    // Transform point cloud to global coordinates
    pcl::PointCloud<pcl::PointXYZ>::Ptr global_pc(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud (*pc, *global_pc, T);
    QueryPointCloudSemantics(octree_ptr, global_pc, pc, csv_file);
  }

}

int main(int argc, char ** argv) {

  ros::init(argc, argv, "nclt_map_node");
  ROS_INFO("nclt 14-class pc_painter init....");
  segmentation_projection::PointCloudPainter<14> painter;

  std::string seg_pcds(argv[1]); // segmented pcds
  std::string pose_path(argv[2]);

  std::ifstream in_pose(pose_path);
  std::string pose_line;
  int counter = 0;
   
  while (std::getline(in_pose, pose_line)) {

    std::stringstream ss(pose_line);    
    uint64_t pose_time;
    ss >> pose_time;

    
    std::string file =  seg_pcds + "/" + std::to_string(pose_time) + ".txt" ;
    if ( !exists( file ) ) {
      std::cout << "Can't find pcd file "<<file<<" at time "<<pose_time << std::endl;
      continue;
    } else
      std::cout<<"Read "<<file<<" at time "<<pose_time << std::endl;

    pcl::PointCloud<pcl::PointSegmentedDistribution<14>>::Ptr pc = read_seg_pcd_text<14>(file);
    //pcl::PointCloud<pcl::PointSegmentedDistribution<14>>::Ptr pc(new pcl::PointCloud<pcl::PointSegmentedDistribution<14>>);
    //pcl::io::loadPCDFile<pcl::PointSegmentedDistribution<14>> (file, *pc);    

    Eigen::Matrix4d T;
    T.setIdentity();
    
    for (int r = 0; r < 3; r++){
      for (int c = 0; c < 4; c++) {
        double pose;
        ss >> pose;
        T(r, c) = pose;
      }
    }


    //for gloabl frame points
    //Eigen::Matrix4d T_inv;
    //T_inv = T.inverse();
    //Eigen::Affine3d T_inv_aff(T_inv.matrix());
    //pcl::transformPointCloud(*pc, *pc, T_inv_aff);
    
    Eigen::Affine3d aff(T.matrix());
    std::cout<<"Process "<<pc->size()<<" labeled points at time "<< pose_time<<std::endl;
    std::cout<<T<<std::endl;

                  
    painter.FuseMapIncremental(*pc, aff, ((double)pose_time) / 1e6, false);
    std::cout<<"counter "<<counter<<std::endl;
    counter ++;

      
  }
  
  query_test(painter.get_octree_ptr());
  

  //for (int i = 0; i!= 14; i++) {
  //  std::cout<<pc[0].label_distribution[i]<<std::endl;
  //}

  ros::spin();

  return 0;
}
