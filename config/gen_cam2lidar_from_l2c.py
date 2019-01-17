
import numpy as np
import cv2, sys

cam2lidar_path = sys.argv[1]

# roll
def Rx(radian):
    M = np.zeros((4,4))
    M[0,0] =  1;
    M[1,1] =  np.cos(radian)
    M[1,2] = -np.sin(radian)
    M[2,1] =  np.sin(radian)
    M[2,2] =  np.cos(radian)
    M[3,3] =  1
    return M

# pitch
def Ry( radian):
    M = np.zeros((4,4))
    M[0,0] =  np.cos(radian)
    M[1,1] =  1
    M[0,2] =  np.sin(radian)
    M[2,0] = -np.sin(radian)
    M[2,2] =  np.cos(radian)
    M[3,3] =  1

    return M

# yawn
def Rz( radian):
    M = np.identity(4)
    M[0,0] =  np.cos(radian)
    M[0,1] = -np.sin(radian)
    M[1,0] =  np.sin(radian)
    M[1,1] =  np.cos(radian)
    M[2,2] =  1
    M[3,3] =  1
    return M




# The one that we measured for realsense2lidar
#c2l_vec = np.array([0.19,  0.04, 0.2, -0.0124 * np.pi / 180, -89.5533* np.pi/180, 91.2846*np.pi/180])
# The one we actually used in this projection package: I tuned the numbers based on 
# my eyes' judgement of projection quality!


l2c_v = np.array([0.041862, -0.001905, -0.000212, 160.868615* np.pi / 180, 89.914152* np.pi / 180, 160.619894* np.pi / 180])
rot = np.matmul(np.matmul(Rz(l2c_v[5]), Ry(l2c_v[4]) ), Rx(l2c_v[3]) )
T_l2c = np.identity(4)
T_l2c[:3,:3] = rot[:3, :3]
T_l2c[:3, 3] = np.transpose(l2c_v[:3])
np.save(cam2lidar_path, np.linalg.inv(T_l2c))

