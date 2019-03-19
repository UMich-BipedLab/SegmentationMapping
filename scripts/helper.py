#!/usr/bin/python


# source: https://github.com/UMich-BipedLab/segmentation_projection
# maintainer: Ray Zhang    rzh@umich.edu



import numpy as np


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
