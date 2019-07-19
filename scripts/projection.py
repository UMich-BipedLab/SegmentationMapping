#!/usr/bin/python


# source: https://github.com/UMich-BipedLab/segmentation_projection
# maintainer: Ray Zhang    rzh@umich.edu



import sys,os, pdb
import numpy as np
import tensorflow as tf
import cv2
from collections import namedtuple
import rospy, pdb

from label2color import  background, label_to_color
from helper import get_cropped_uv_rotated, is_out_of_bound, is_out_of_bound_rotated, softmax, publish_pcl_pc2, publish_pcl_pc2_label

from NeuralNetConfigs import NeuralNetConfigs

class LidarSeg:
    def __init__(self, net_configs):

        # read the network
        with tf.gfile.GFile(net_configs.path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        # tf configs
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)

        # the input and output of the neural net
        self.y, = tf.import_graph_def(graph_def, return_elements=[net_configs.label_output_tensor])
        self.G = tf.get_default_graph()
        self.x = self.G.get_tensor_by_name(net_configs.image_input_tensor)
        if net_configs.is_train_input_tensor is not None:            
            self.is_train_input_tensor = self.G.get_tensor_by_name(net_configs.is_train_input_tensor)
        else:
            self.is_train_input_tensor = None
        if net_configs.distribution_output_tensor is not None:
            self.distribution_tensor = self.G.get_tensor_by_name(net_configs.distribution_output_tensor)
        else:
            self.distribution_tensor = None
        self.num_output_class = net_configs.num_classes
        pdb.set_trace()
        # initialization
        tf.global_variables_initializer().run(session=self.sess)

        self.intrinsic = []
        self.cam2lidar = []
        self.distort_map = []
        self.counter = 0

    def add_cam(self,intrinsic_mat, cam2lidar, distort):
        """
        param: intrinsic_mat: 3x4 
               extrinsic_mat: 4x4, 
               distort: pixelwise map for distortion
        """
        self.intrinsic.append(intrinsic_mat)
        self.cam2lidar.append( cam2lidar )
        self.distort_map.append(distort)

        

    def project_lidar_to_seg(self, lidar, rgb_img, camera_ind,  original_img):
        """
        assume the lidar points can all be projected to this img
        assume the rgb img shape meets the requirement of the neural net input

        lidar points: 3xN
        """
        if self.distribution_tensor is not None:
            if self.is_train_input_tensor is not None:
                distribution = self.sess.run([self.distribution_tensor],
                                             feed_dict={self.x: np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB),axis=0),
                                                        self.is_train_input_tensor: False})[0]
            else:
                distribution = self.sess.run([self.distribution_tensor],
                                             feed_dict={self.x: np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB),axis=0)})[0]
            
            distribution = distribution[0, :, :, :]
        else:
            if self.is_train_input_tensor is not None:
                out = self.sess.run(self.y,
                                    feed_dict={self.x: np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB),axis=0),
                                               self.is_train_input_tensor: False})
            else:
                out = self.sess.run(self.y,
                                    feed_dict={self.x: np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB),axis=0)})
                
            distribution = []
            out = out [0,:,:]


        # project lidar points into camera coordinates
        T_c2l = self.cam2lidar[camera_ind][:3, :]
        lidar_in_cam = np.matmul(self.intrinsic[camera_ind], T_c2l )
        projected_lidar_2d = np.matmul( lidar_in_cam, lidar)
        projected_lidar_2d[0, :] = projected_lidar_2d[0, :] / projected_lidar_2d[2, :]
        projected_lidar_2d[1, :] = projected_lidar_2d[1, :] / projected_lidar_2d[2, :]
        #print("total number of lidar : "+str(projected_lidar_2d.shape))
        idx_infront = projected_lidar_2d[2, :]>0
        print("idx_front sum is " + str(np.sum(idx_infront)))
        x_im = projected_lidar_2d[0, :][idx_infront]
        y_im = projected_lidar_2d[1, :][idx_infront]
        z_im = projected_lidar_2d[2, :][idx_infront]
        points_on_img = np.zeros((3, x_im.size))
        points_on_img[0, :] = x_im
        points_on_img[1, :] = y_im
        points_on_img[2, :] = z_im
        lidar_on_img = np.zeros((3, x_im.size))
        lidar_on_img[0, :] = lidar[0, :][idx_infront]
        lidar_on_img[1, :] = lidar[1, :][idx_infront]
        lidar_on_img[2, :] = lidar[2, :][idx_infront]
                
        # distort the lidar points based on the distortion map file
        projected_lidar_2d, remaining_ind = self.distort_map[camera_ind].distort(points_on_img)
        lidar_on_img = lidar_on_img[:, remaining_ind]
        print(projected_lidar_2d.shape, lidar_on_img.shape)
        #######################################################################
        # for debug use: visualize the projection on the original rgb image
        #######################################################################
        #for col in range(projected_lidar_2d.shape[1]):
        #    u = int(round(projected_lidar_2d[0, col] , 0))
        #    v = int(round(projected_lidar_2d[1, col] , 0))
        #    cv2.circle(original_img, (u, v),2, (0,0,255))
        #cv2.imwrite("original_project"+str(self.counter)+".png", original_img)
        #cv2.imshow("original projection", original_img)
        #cv2.waitKey(100)
        #print("write projection")
        #cv2.imwrite("labels_"+str(self.counter)+".png", out)
        ########################################################################
        
        projected_points = []
        projected_index  = []  # just for visualization
        labels = []
	original_rgb = []
        class_distribution = []
        for col in range(projected_lidar_2d.shape[1]):
            u, v, _ = projected_lidar_2d[:, col]
            u ,v = get_cropped_uv_rotated(u, v, rgb_img.shape[1] * 1.0 / 1200 )
            if is_out_of_bound(u, v, rgb_img.shape[1], rgb_img.shape[0]):
                continue
            projected_points.append(lidar_on_img[:, col])
            if self.distribution_tensor is not None:
                distribution_normalized = softmax(distribution[int(v), int(u), :])
                distribution_normalized[0] += np.sum(distribution_normalized[self.num_output_class:])
                distribution_normalized = distribution_normalized[:self.num_output_class]
                class_distribution.append(distribution_normalized)
                label = np.argmax(distribution_normalized)
            else:
                label = out[int(v), int(u)] if out[v, u] < self.num_output_class else 0
            labels.append(label)
	    original_rgb.append(rgb_img[int(v), int(u), :])
            projected_index.append(col)
        print(" shape of projected points from camera # " + str(camera_ind)+ " is "+str(points_on_img.shape)+", # of in range points is "+str(len(labels)))
        ##########################################################################
        # uncomment this if you want to visualize the projection && labeling result
        self.visualization(labels, projected_index, projected_lidar_2d, rgb_img)
        self.counter +=1
        ##########################################################################
        #publish_pcl_pc2_label(projected_points, labels )
        return labels, projected_points, class_distribution, original_rgb
        

    def visualization(self, labels,index,  projected_points_2d, rgb_img):
        to_show = rgb_img
        #print("num of projected lidar points is "+str(len(labels)))
        for i in range(len(labels )):
            p = (int(projected_points_2d[0, index[i]]),
                 int(projected_points_2d[1, index[i]]))
            #cv2.putText(to_show,. str(int(labels[i])), 
            #            p,
            #            cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,203))
            #if is_out_of_bound(p[0], p[1]): continue
            
            if labels[i] in label_to_color:
                color = label_to_color[labels[i]]
            else:
                color = label_to_color[background]

            cv2.circle(to_show,get_cropped_uv_rotated(p[0], p[1], 512/600.0),2, (color[2], color[1], color[0] ))
        #cv2.imwrite("segmentation_projection_out/projected"+str(self.counter)+".png", to_show)

        cv2.imshow("projected",to_show)
        cv2.waitKey(100)

        


    
    
