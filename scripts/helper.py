#!/usr/bin/python


# source: https://github.com/UMich-BipedLab/segmentation_projection
# maintainer: Ray Zhang    rzh@umich.edu



import numpy as np

import pcl  # python-pcl
import pcl.pcl_visualization
from label2color import label_to_color

def publish_pcl_pc2_label(lidar, labels):
    '''
    for debugging: visualizing the constructed labeled pointcloud before
                    we publish them
    lidar: a python array
    '''
    cloud = pcl.PointCloud_PointXYZRGB()
    points = np.zeros((len(lidar), 4), dtype=np.float32)
    for i in range(len(lidar)):
        # set Point Plane
        points[i][0] = lidar[i][0]
        points[i][1] = lidar[i][1]
        points[i][2] = lidar[i][2]
        color = label_to_color[labels[i]]
        points[i][3] = int(color[0]) << 16 | int(color[1]) << 8 | int(color[2])
    cloud.from_array(points)
    pcl_vis = pcl.pcl_visualization.CloudViewing()
    pcl_vis.ShowColorCloud(cloud)
    v = True
    while v:
        v=not(pcl_vis.WasStopped())

def publish_pcl_pc2(lidar):
    '''
    for debugging: visualizing the constructed labeled pointcloud before
                    we publish them
    lidar: a 3xN numpy array
    '''
    print("visualizing uncolored pcl")
    cloud = pcl.PointCloud_PointXYZRGB()
    points = np.zeros((lidar.shape[1], 4), dtype=np.float32)
    for i in range(lidar.shape[1] ):
        # set Point Plane
        points[i][0] = lidar[0, i]
        points[i][1] = lidar[1, i]
        points[i][2] = lidar[2, i]
        points[i][3] = 0 << 16 | 255 << 8 | 0 
    cloud.from_array(points)
    pcl_vis = pcl.pcl_visualization.CloudViewing()
    pcl_vis.ShowColorCloud(cloud)
    v = True
    while v:
        v=not(pcl_vis.WasStopped())

def get_cropped_uv_rotated( u, v, small_over_large_scale):
    u = u - 500
    v = 1200 - 1 - v
    u = int(round(u * small_over_large_scale, 0))
    v = int(round(v * small_over_large_scale, 0))
    u_new = v
    v_new = u
    return u_new,v_new

def is_out_of_bound(u, v, max_u, max_v):
    #if u < 500 or u >= 1100 or \
    #   v < 0   or v >= 1200:
    if u >= max_u or u < 0 or \
       v >= max_v or v < 0:
        return True
    else:
        return False 

def is_out_of_bound_rotated(u, v):
    if u < 0 or u > 1200 or \
       v< 500 or v > 1100 :
        return True
    else:
        return False
    

def softmax(dist_arr):
    exps = np.exp(dist_arr)
    return exps/np.sum(exps)

def softmax_img(dist_arr):
    exps = np.exp(dist_arr)
    return exps/np.sum(exps)
