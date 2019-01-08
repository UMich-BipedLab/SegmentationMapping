import sys,os
import numpy as np
#import tensorflow as tf
import cv2

import rospy


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


    def add_cam(self,instrinsic_mat, cam2lidar):
        """
        param: intrinsic_mat: 3x4 
               extrinsic_mat: 4x4, imu2cam
        """
        self.intrinsic.append(intrinsic_mat)
        self.cam2lidar.append( cam2lidar )

    def project_lidar_to_seg(self, lidar, rgb_img, camera_ind):
        """
        assume the lidar points can all be projected to this img
        assume the rgb img shape meets the requirement of the neural net

        lidar points: 3xN
        """
        out = self.sess.run(self.y, feed_dict={self.x: rgb_img})
        T_c2l = self.cam2lidar[camera_ind]
        lidar_in_cam = np.matmul(T_c2l, lidar)
        projected_lidar_2d = np.matmul(self.intrinsic[camera_ind][:, :-1], lidar_in_cam)
        projected_lidar_2d[:, 0] = projected_lidar_2d[:, 0] / projected_lidar_2d[:,2]
        projected_lidar_2d[:, 1] = projected_lidar_2d[:, 1] / projected_lidar_2d[:,2]
        projected_lidar_2d[:, 2] = 1
        labels = np.zeros((1, lidar.shape[1] ))

        for col in range(projected_lidar_2d.shape[1]):
            u, v, d = projected_lidar_2d[:, col]
            print("coordinate "+str((u,v,d)))
            labels[col] = out[u, v]

        self.visualization(labels, projected_lidar_2d, rgb_img)
        return labels
        

    def visualization(self, labels, projected_points, rgb_img):
        to_show = rgb_img
        for i in range(labels.shape[1] ):
            cv2.putText(to_show, str(labels[i]), 
                        (projected_points[1, i], projected_points[0,i]),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.imshow("projected",to_show)
        cv2.waitKey(0)

        


    
    
