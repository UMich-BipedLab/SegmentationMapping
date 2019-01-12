import sys,os, pdb
import numpy as np
import tensorflow as tf
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
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # the input and output of the neural net
        self.y, = tf.import_graph_def(graph_def, return_elements=['network/output/ArgMax:0'])
        self.G = tf.get_default_graph()
        self.x = self.G.get_tensor_by_name('import/network/input/Placeholder:0')
        self.is_train = self.G.get_tensor_by_name('import/network/input/Placeholder_2:0')

        # initialization
        tf.global_variables_initializer().run(session=self.sess)

        self.intrinsic = []
        self.cam2lidar  = []


    def add_cam(self,intrinsic_mat, cam2lidar):
        """
        param: intrinsic_mat: 3x4 
               extrinsic_mat: 4x4, imu2cam
        """
        self.intrinsic.append(intrinsic_mat)
        self.cam2lidar.append( np.linalg.inv(cam2lidar ))
        print("add camera: intrinsic ")
        print(intrinsic_mat)
        print("add camera: extrinsic ")
        print(cam2lidar)

    def project_lidar_to_seg(self, lidar, rgb_img, camera_ind):
        """
        assume the lidar points can all be projected to this img
        assume the rgb img shape meets the requirement of the neural net

        lidar points: 3xN
        """
        out = self.sess.run(self.y,
                            feed_dict={self.x: np.expand_dims(rgb_img,axis=0),
                                       self.is_train: False})[0, :, :]

        T_c2l = self.cam2lidar[camera_ind]
        lidar_in_cam = np.matmul(self.intrinsic[camera_ind], T_c2l )
        projected_lidar_2d = np.matmul( lidar_in_cam, lidar)
        projected_lidar_2d[0, :] = projected_lidar_2d[0, :] / projected_lidar_2d[2, :]
        projected_lidar_2d[1, :] = projected_lidar_2d[1, :] / projected_lidar_2d[2, :]
        projected_lidar_2d[2, :] = 1

        projected_points = []
        projected_index  = []
        labels = []

        for col in range(projected_lidar_2d.shape[1]):
            u, v, d = projected_lidar_2d[:, col]
            if u < 0 or u > rgb_img.shape[1] or v < 0 or v > rgb_img.shape[0]:
                continue
            #print("coordinate "+str((u,v,d)) )
            projected_points.append(lidar[:, col])
            labels.append(out[int(v), int(u)])
            projected_index.append(col)

        self.visualization(labels, projected_index, projected_lidar_2d, rgb_img)
        return labels, projected_points
        

    def visualization(self, labels,index,  projected_points_2d, rgb_img):
        to_show = rgb_img
        print("num of projected lidar points is "+str(len(labels)))
        for i in range(len(labels )):
            p = (int(projected_points_2d[0, index[i]]),
                 int(projected_points_2d[1, index[i]]))
            #cv2.putText(to_show, str(int(labels[i])), 
            #            p,
            #            cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,203))
            cv2.circle(to_show,p,2, (0,0,203))
        cv2.imwrite("projected.png", to_show)
        #cv2.imshow("projected",to_show)
        # cv2.waitKey(0)

        


    
    
