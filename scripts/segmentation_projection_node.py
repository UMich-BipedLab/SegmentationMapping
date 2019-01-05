
# ros
import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud

# our projection library
from projection import LidarSeg


class SegmentationProjectionNode:
    def __init__():
        # buffer 
        self.img_list = []
        self.pc_list  = []
        
        rospy.init_node('segmenation_projection_node', anonymous=True)
        rospy.sleep(0.5)
        
        # the segmentation object
        self.lidar_seg = LidarSeg(rospy.get_param('neural_net_graph'))

        # labeled pointcloud publisher
        label_pc_topic = rospy.get_param('labeled_pointcloud')
        self.labeled_pointcloud_publisher = rospy.Publisher(label_pc_topic , PointCloud)

        # lidar subscription
        lidar_topic = rospy.get_param('lidar')
        lidar_sub   = message_filters.Subscriber(lidar_topic, PointCloud2)

        # multiple camera subscription
        camera_num  = rospy.get_param('camera_num')
        image_topic_list = []
        sub_list   = [lidar_sub]

        for i in range(camera_num):
            image_topic_list.append( rospy.get_param('image_'+str(i)) )
            cam2lidar_file = rospy.get_param('cam2lidar_file_'+str(i))
            intrinsic_file = rospy.get_param('cam_instrinsic_file_'+str(i))
            T_cam2lidar = np.load(cam2lidar_file)
            lidar_seg.add_cam(intrinsic_file, T_cam2lidar)
            sub_list.append(message_filters.Subscriber(image_topic_list[i], Image))
        
        # construct message filter
        ts = message_filters.TimeSynchronizer(sub_list, 50)
        ts.registerCallback(self.callback)
 

    def lidar_msg_to_pointcloud(original_pc):
        lidar = np.zeros((3, len(original_pc.points)))
        ind = 0
        for p in original_pc.points:
            lidar[0, ind] = p.x
            lidar[1, ind] = p.y
            liadr[2, ind] = p.z
            ind += 1
        return lidar
        

    def publish_pointcloud(original_pc, labels):
        #declaring pointcloud
        to_publish = PointCloud()
        #filling pointcloud header
        to_publish = original_pc
        to_publish.channels.name   = "labels"
        to_publish.channels.values = labels.tolist()
        #publish
        print ("publishing pointcloud with labels")
        self.labeled_pointcloud_publisher.publish(to_publish)



    def callback(sub_list):
        
        img_msg_list = sub_list[:-1]
        lidar_msg    = sub_list[-1]
        
        lidar = lidar_msg_to_pointcloud(lidar_msg)
        labeled_points = self.lidar_seg.project_lidar_to_seg(lidar, img_list)
        publish_pointcloud(lidar_msg, labeled_points)



    def main():
        rospy.spin()
    


if __name__ == "__main__":

    node = SegmentationProjectionNode()
    node.main()


