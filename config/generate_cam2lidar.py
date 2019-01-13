
import numpy as np
import cv2

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
c2l_vec = np.array([0.19,  0.04, 0.2, -1.5 * np.pi / 180, -81* np.pi/180, 93.2846*np.pi/180])

rot = np.matmul(np.matmul(Rz(c2l_vec[5]), Ry(c2l_vec[4]) ), Rx(c2l_vec[3]) )
T_c2l = np.identity(4)
T_c2l[:3,:3] = rot[:3, :3]
T_c2l[:3, 3] = np.transpose(c2l_vec[:3])
np.save("realsense2lidar.npy", T_c2l)

