#!/usr/bin/python

import sys, os, pdb
import numpy as np
import cv2
import argparse
from distort import DistortMap
from helper import get_cropped_uv_rotated, is_out_of_bound, is_out_of_bound_rotated
from label2color import  background, label_to_color
from pcl import pcl_visualization
import pcl

def gt_viewer(lidar_final, labels_final):
    lidar = np.transpose(lidar_final).astype(np.float32)
    cloud = pcl.PointCloud_PointXYZRGB()
    for i in range(len(labels_final) ):
        # set Point Plane
        if labels_final[i] in label_to_color:
            color = label_to_color[labels_final[i]]
        else:
            color = label_to_color[0]
        
        lidar[i][3] = color[2] << 16 | color[1] << 8 | color[0]
    cloud.from_array(lidar)
    visual = pcl_visualization.CloudViewing()
    visual.ShowColorCloud(cloud)

    v = True
    while v:
        v=not(visual.WasStopped())

def gt_projection(lidar, lidar_distribution,
                  rgb_img_shape, gt_img, cam2lidar, intrinsic, distort_map, output_file):

    num_classes = lidar_distribution.shape[0]
    
    # project lidar points into camera coordinates
    T_c2l = cam2lidar[:3, :]

    lidar_in_cam = np.matmul(intrinsic, T_c2l )
    projected_lidar_2d = np.matmul( lidar_in_cam, lidar)
    projected_lidar_2d[0, :] = projected_lidar_2d[0, :] / projected_lidar_2d[2, :]
    projected_lidar_2d[1, :] = projected_lidar_2d[1, :] / projected_lidar_2d[2, :]

    # filter the points on the image
    idx_infront = projected_lidar_2d[2, :]>0
    print("idx_front sum is " + str(np.sum(idx_infront)))
    filtered_distribution = lidar_distribution[:, idx_infront]
    points_on_img = projected_lidar_2d[:, idx_infront]
    lidar_on_img = lidar[:, idx_infront]

                
    # distort the lidar points based on the distortion map file
    projected_lidar_2d, remaining_ind = distort_map.distort(points_on_img)
    lidar_on_img = lidar_on_img[:, remaining_ind]
    filtered_distribution = filtered_distribution[:, remaining_ind]
    print(projected_lidar_2d.shape, lidar_on_img.shape, filtered_distribution.shape)
        
    projected_points = []
    projected_index  = []  # just for visualization
    labels = []
    #original_rgb = []
    class_distribution = []
    gt_label_file = open(output_file, 'w')
    for col in range(projected_lidar_2d.shape[1]):
        u, v, _ = projected_lidar_2d[:, col]
        u ,v = get_cropped_uv_rotated(u, v, rgb_img_shape[1] * 1.0 / 1200 )
        if is_out_of_bound(u, v, rgb_img_shape[1], rgb_img_shape[0]):
            continue
        projected_points.append(lidar_on_img[:, col])

        if gt_img[v, u] < num_classes :
            label = gt_img[int(v), int(u)]
        else:
            label = 0
        labels.append(label)

        # write gt results to the file
        gt_label_file.write("{} {} {} ".format(lidar_on_img[0, col], lidar_on_img[1, col], lidar_on_img[2, col]))
        for i in range(num_classes):
            gt_label_file.write("{} ".format(filtered_distribution[i, col]))
        gt_label_file.write(str(label))
        gt_label_file.write("\n")
        
	#original_rgb.append(rgb_img[int(v), int(u), :])
        projected_index.append(col)
    gt_label_file.close()
    print("Finish writing to "+output_file)

    lidars_left = lidar_on_img[:, projected_index]
    distribution_left = filtered_distribution[:, projected_index]
    #######################################################################
    # for debug use: visualize the projection on the original rgb image
    #######################################################################
    img_to_vis = np.zeros((gt_img.shape[0], gt_img.shape[1], 3))
    img_to_vis[:,:,0] = gt_img
    img_to_vis[:,:,1] = gt_img
    img_to_vis[:,:,2] = gt_img
    for j in range(len(projected_index)):
        col = projected_index[j]
        u = int(round(projected_lidar_2d[0, col] , 0))
        v = int(round(projected_lidar_2d[1, col] , 0))
        u ,v = get_cropped_uv_rotated(u, v, rgb_img_shape[1] * 1.0 / 1200 )
        if labels[j] in label_to_color:
            color = label_to_color[labels[j]]
        else:
            color = label_to_color[0]

        cv2.circle(img_to_vis, (u, v),2, color)
    cv2.imshow("gt projection", img_to_vis)
    cv2.waitKey(1000)
    ########################################################################
    return lidars_left, labels, distribution_left


def read_input_pc_with_distribution(file_name):
    lidar = []
    lidar_distribution = []
    with open(file_name) as f:
        for line in f:
            point = line.split()
            point_np = np.array([float(item) for item in point])

            lidar.append(np.array([point_np[0],point_np[1],point_np[2], 1]))
            
            lidar_distribution.append(point_np[3:])

    lidar = np.transpose(np.array(lidar))
    lidar_distribution = np.transpose(np.array(lidar_distribution))
    print("Finish reading data... lidar shape is {}, lidar_distribution shape is {}".format(lidar.shape, lidar_distribution.shape))
    return lidar, lidar_distribution


def batch_gt_projection_nclt(query_lidar_folder,
                             raw_rgb_img_shape,
                             gt_folder,
                             config_folder,
                             output_folder):
    cam2lidar_mats = []
    distortions    = []
    intrinsic      = []
    for i in range(5):
        cam2lidar_mats.append(np.load("{}/cam2lidar_{}.npy".format(config_folder, i+1)))
        intrinsic.append("{}/nclt_intrinsic_{}.npy".format(config_folder, i+1))
        distortions.append(DistortMap("{}/U2D_Cam{}_1616X1232.txt".format(config, i+1)))
    
    for query_file in sorted(os.listdir(gt_folder)):

        camera_prefix = query_file[:4]
        camera_ind = int(camera_prefix[-1])
        camera_i_list = camera_ind - 1
        time = query_file[5:-4]
        print("Processing "+query_file+", time is "+time)
        
        lidar, lidar_distribution = read_input_pc_with_distribution( query_lidar_folder + "/" + time + ".txt" )

        gt = cv2.imread( gt_folder + "/" + query_file )[:,:,0]
        print(lidar.shape, lidar_distribution.shape, gt.shape)
        lidars_left, labels, distribution_left = gt_projection(lidar,
                                                               lidar_distribution,
                                                               raw_rgb_img_shape,
                                                               gt,
                                                               cam2lidar_mats[camera_i_list],
                                                               intrinsic[camera_i_list],
                                                               distort[camera_i_list],
                                                               output_folder+"/"+camera_prefix + "_"+ time+".txt" )
        
        gt_viewer(lidars_left, labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--segmented_lidar_folder',  type=str, nargs=1)
    parser.add_argument('--gt_folder', type=str, nargs=1)
    parser.add_argument('--config_folder', type=str, nargs=1)
    parser.add_argument('--output_folder', type=str, nargs=1)
    parser.add_argument('--rgb_img_shape', type=int, nargs=2)

    args = parser.parse_args()


    batch_gt_projection_nclt(args.segmented_lidar_folder,
                             args.rgb_img_shape,
                             args.gt_folder,
                             args.config_folder,
                             args.output_folder)
