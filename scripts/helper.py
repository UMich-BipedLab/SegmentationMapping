

import numpy as np


def get_cropped_uv_rotated( u, v):
    u = u - 500
    v = 1200 - 1 - v
    u = int(round(u * 2.0 / 3.0, 0))
    v = int(round(v * 2.0 / 3.0, 0))
    u_new = v
    v_new = u
    return u_new,v_new

def is_out_of_bound(u, v):
    #if u < 500 or u >= 1100 or \
    #   v < 0   or v >= 1200:
    if u >= 800 or u < 0 or \
       v >= 400 or v < 0:
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
