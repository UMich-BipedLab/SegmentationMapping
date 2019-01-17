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

class Undistort(object):

    def __init__(self, fin, scale=1.0, fmask=None):
        self.fin = fin
        # read in distort
        with open(fin, 'r') as f:
            #chunks = f.readline().rstrip().split(' ')
            header = f.readline().rstrip()
            chunks = re.sub(r'[^0-9,]', '', header).split(',')
            self.mapu = np.zeros((int(chunks[1]),int(chunks[0])),
                    dtype=np.float32)
            self.mapv = np.zeros((int(chunks[1]),int(chunks[0])),
                    dtype=np.float32)
            for line in f.readlines():
                chunks = line.rstrip().split(' ')
                self.mapu[int(chunks[0]),int(chunks[1])] = float(chunks[3])
                self.mapv[int(chunks[0]),int(chunks[1])] = float(chunks[2])
        # generate a mask
        self.mask = np.ones(self.mapu.shape, dtype=np.uint8)
        self.mask = cv2.remap(self.mask, self.mapu, self.mapv, cv2.INTER_LINEAR)
        kernel = np.ones((30,30),np.uint8)
        self.mask = cv2.erode(self.mask, kernel, iterations=1)

    """
    Optionally, define a mask
    """
    def set_mask(fmask):
        # add in the additional mask passed in as fmask
        if fmask:
            mask = cv2.cvtColor(cv2.imread(fmask), cv2.COLOR_BGR2GRAY)
            self.mask = self.mask & mask
        new_shape = (int(self.mask.shape[1]*scale), int(self.mask.shape[0]*scale))
        self.mask = cv2.resize(self.mask, new_shape,
                               interpolation=cv2.INTER_CUBIC)
        #plt.figure(1)
        #plt.imshow(self.mask, cmap='gray')
        #plt.show()

    """
    Use OpenCV to undistorted the given image
    """
    def undistort(self, img):
        return cv2.resize(cv2.remap(img, self.mapu, self.mapv, cv2.INTER_LINEAR),
                          (self.mask.shape[1], self.mask.shape[0]),
                          interpolation=cv2.INTER_CUBIC)

def main():
    parser = argparse.ArgumentParser(description="Undistort images")
    parser.add_argument('image', metavar='img', type=str, help='image to undistort')
    parser.add_argument('map', metavar='map', type=str, help='undistortion map')

    args = parser.parse_args()

    undistort = Undistort(args.map)
    print 'Loaded camera calibration'

    im = cv2.imread(args.image)
    print(im.shape)
    #cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    #cv2.imshow('Image', im)

    im_undistorted = undistort.undistort(im)

    #cv2.namedWindow('Undistorted Image', cv2.WINDOW_NORMAL)
    #cv2.imshow('Undistorted Image', im_undistorted)
    cv2.imwrite("undist.png", im_undistorted)  
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
