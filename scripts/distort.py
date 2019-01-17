"""
Demonstrating how to undistort images.

Reads in the given calibration file, parses it, and uses it to undistort the given
image. Then display both the original and undistorted images.

To use:

    python undistort.py image calibration_file
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import re

class DistortMap(object):

    def __init__(self, undist2distorted_map, scale=1.0, fmask=None):
        # undist2distorted_map example:  D2U_Cam1_1616X1232.txt 
        # read in distort
        
        with open(undist2distorted_map, 'r') as f:
            #chunks = f.readline().rstrip().split(' ')
            header = f.readline().rstrip()

            # chunks[0]: width    chunks[1]: height
            chunks = re.sub(r'[^0-9,]', '', header).split(',')  
            self.mapu = np.zeros((int(chunks[1]),int(chunks[0])),
                    dtype=np.float32)
            self.mapv = np.zeros((int(chunks[1]),int(chunks[0])),
                    dtype=np.float32)

            # undistorted lidar -> distorted camera index
            # [v_projected_lidar, u_projected_lidar] --- >  (v_cam, u_cam)
            for line in f.readlines():
                chunks = line.rstrip().split(' ') 
                self.mapu[int(chunks[0]),int(chunks[1])] = float(chunks[3])
                self.mapv[int(chunks[0]),int(chunks[1])] = float(chunks[2])

                
    def distort(self, lidar_projected_2d):
        '''
        lidar_projected_2d:  3*N np array. The last row contains only 1s
        '''
        distorted = []#np.ones(lidar_projected_2d.shape)
        counter = 0
        for col in range(lidar_projected_2d.shape[1]):
            u = int(lidar_projected_2d[0, col])
            v = int(lidar_projected_2d[1, col])
            if u < 0 or u > 1200 or v< 500 or v > 1100:
                continue
            dist = np.ones((1,3))
            dist[0] = self.mapu[u, v]
            dist[0,1] = self.mapv[u, v]
            distort.append(dist)

        return np.array(distorted).transpose()



def main():
    parser = argparse.ArgumentParser(description="Undistort images")
    parser.add_argument('image', metavar='img', type=str, help='image to undistort')
    parser.add_argument('map', metavar='map', type=str, help='undistortion map')

    args = parser.parse_args()

    distort = DistortMap(args.map)
    print 'Loaded camera calibration'



    #cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
    #cv2.imshow('Undistorted Image', im_undistorted)
    cv2.imwrite("undist.png", im_undistorted)  
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
