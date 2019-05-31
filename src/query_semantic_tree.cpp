// 
// Query semantic tree for evaluation
// Lu Gan, ganlu@umich.edu
//

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <boost/filesystem.hpp>

#include <pcl/common/transforms.h>

#include <octomap/SemanticOcTree.h>

#include "PointSegmentedDistribution.hpp"

using namespace octomap;
namespace fs = boost::filesystem;

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
  


void QueryPointCloudSemantics(
    const SemanticOcTree* tree,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
    std::string file_name) {

    std::string txt_file = file_name.substr(0, file_name.size()-8) + ".txt";
    std::cout << txt_file << std::endl;
    std::ofstream data_out(txt_file);

    std::cout << "Performing some queries ..." << std::endl;
    for (int j = 0; j < pc->points.size(); ++j) {
      point3d query (pc->points[j].x, pc->points[j].y, pc->points[j].z);
      SemanticOcTreeNode* node = tree->search(query);
      if (node != NULL && node->getOccupancy() > 0.5 && node->isSemanticsSet()) {
          //PrintQueryInfo(query, node);
          std::cout << node->isSemanticsSet() << std::endl;
          auto semantics = node->getSemantics();
          std::cout << semantics.label.size() << std::endl;
          data_out << pc->points[j].x << " "
                   << pc->points[j].y << " "
                   << pc->points[j].z;
          
          for (int c = 0; c < semantics.label.size(); ++c)
            data_out << " " << semantics.label[c];
          data_out.close();
      } else
        continue;
    }
}


std::unordered_map<uint64_t, std::vector<double>> ReadPointCloudPoses(std::string pose_file) {
  std::ifstream pose_in(pose_file);
  std::string pose_line;
  std::unordered_map<uint64_t, std::vector<double>> time2pose;
  
  while (std::getline(pose_in, pose_line)) {
    std::stringstream ss(pose_line);
    uint64_t t;
    ss >> t;
    uint64_t time = (uint64_t) (std::round((double)t / 1000.0) + 0.1);
    
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




int main(int argc, char** argv) {

  // Read semantic octree from tree file
  std::string tree_file = "/home/biped/.ros/prior_octree_with_distribution.ot";
  std::cout << "Read semantic tree from " << tree_file << "\n";
  AbstractOcTree* read_tree = AbstractOcTree::read(tree_file);
  SemanticOcTree* read_semantic_tree = dynamic_cast<SemanticOcTree*> (read_tree);

  // Read pc poses
  std::string pose_file = "/home/biped/perl_code/workspace/src/segmentation_projection/data/nclt_04_29/pose_sequence.txt";
  std::unordered_map<uint64_t, std::vector<double>> time2pose = ReadPointCloudPoses(pose_file);
  std::cout << "Read poses size: " << time2pose.size() << std::endl;

  // Read gt times
  std::string time_file = "/home/biped/perl_code/rvsm/nclt_may23_gt/gt_times.txt";
  std::ifstream time_in(time_file);
  std::string time_line;
  std::string seq_path("/home/biped//perl_code/gicp/visualization/data/rvm_04_29/");
  
  while (std::getline(time_in, time_line)) {
    std::stringstream ss(time_line);
    uint64_t time;
    ss >> time;
    std::string csv_file = seq_path + "/" + std::to_string(time) + ".bin.csv";
    if (!fs::exists(csv_file)) {
      std::cout << "Can't find csv file " << csv_file << " at time " << time << std::endl;
      continue;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr pc = read_nclt_pc_file(csv_file);
    std::cout << "Process " << pc->size() << " points at time " << time << std::endl;
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
    QueryPointCloudSemantics(read_semantic_tree, global_pc, csv_file);
  }
}
  

