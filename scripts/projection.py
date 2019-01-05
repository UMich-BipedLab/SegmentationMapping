
import numpy as np
import tensorflow as tf
import sys,os
import tensorflow as tf

import rospy
import cv_bridge

class LidarSeg:
    def __init__(self, neural_net_graph_path):

        # read the network
        with tf.gfile.GFile(neural_net_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        # tf configs
        config = tf.ConfigProto()
        self.G = tf.Graph()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=G, config=config)

        # the input and output of the neural net
        self.y, = tf.import_graph_def(graph_def, return_elements=['network/output/ArgMax:0'])
        self.x = G.get_tensor_by_name('import/network/input/Placeholder:0')

        # initialization
        tf.global_variables_initializer().run()

        self.instrinsic = []
        self.cam2lidar  = []

        # cv_bridge
        self.bridge = CvBridge()


    def add_lidar(T_imu2lidar):
        self.imu2lidar = T_imu2lidar
    
    def add_cam(instrinsic_mat, imu2cam):
        """
        param: intrinsic_mat: 3x4 
               extrinsic_mat: 4x4, imu2cam
        """
        self.intrinsic.append(intrinsic_mat)
        self.cam2lidar.append(np.linalg.inv(imu2cam) *self.imu2lidar)

    def project_lidar_to_seg(lidar, rgb_img, camera_ind):
        """
        assume the lidar points can all be projected to this img
        lidar points: 3xN
        """
        out = self.sess.run(self.y, feed_dict={self.x: rgb_img})
        T_c2l = self.cam2lidar[camera_ind]
        lidar_in_cam = np.matmul(T_c2l, lidar)
        projected_lidar_2d = np.matmul(self.intrinsic[camera_ind], lidar_in_cam)
        labels = np.zeros((1, ))

        for col in range(projected_lidar_2d.shape[1]):
            u, v, d = projected_lidar_2d[:, col]
            print("coordinate "+str((u,v,d)))
            label = out[u, v]


        return projected_lidar_2d
        

    

if __name__ == "__main__":

    lidar_seg = LidarSeg("graph_optimized_320p.pb")

    
    lidar_seg.add_cam()

    
    
