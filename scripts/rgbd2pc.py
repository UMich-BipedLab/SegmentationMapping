#!/usr/bin/python


# source: https://github.com/UMich-BipedLab/segmentation_projection
# maintainer: Ray Zhang    rzh@umich.edu


# ros
import rospy
import ros_numpy
import message_filters
from sensor_msgs.msg import Image, PointCloud, Pointcloud2, ChannelFloat32
from geometry_msgs.msg import Point32  
from cv_bridge import CvBridge, CvBridgeError

from label2color import label_to_color, background
from helper import publish_pcl_pc2_label

import cv2, time
import numpy as np
from scipy.ndimage import rotate
# debug
import pdb
import os, sys


class RgbdSegmentationNode:
    '''
    This class takes RGBD segmented images and correspoinding depth images, 
    and generates the labeled pointcloud msg
    '''
    def __init__(self):
        rospy.init_node('rgbd_segmentation_node', anonymous=True)
        rospy.sleep(0.5)

        self.is_input_from_file = rospy.get_param("~is_input_from_file")
        if self.is_input_from_file:
            self.segmented_img_folder = rospy.get_param("~segmented_img_folder")
            if not os.path.exists(self.segmented_img_folder):
                print("segmented image folder does not exist: " + self.segmented_img_folder)
                exit(0)
            #self.rgb_img_folder = rospy.get_param("~rgb_img_folder")
            #self.depth_img_folder = rospy.get_param("~depth_img_folder")
        self.is_input_segmented = rospy.get_param("~is_input_segmented")

        labeled_pc_topic = rospy.get_param("~labeled_pointcloud")
        self.labeled_pc_publisher = rospy.Publisher(labeled_pc_topic, PointCloud, queue_size = 40)


    def publish_dense_labeled_pointcloud(self):
        pass

    def generate_labeled_pc_from_scenenet(self):
        trajectories = sorted(os.listdir(self.segmented_img_folder))

        
