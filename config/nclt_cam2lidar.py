
import numpy as np
import cv2, sys



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

# r, p, y is in degree
def sixdof_to_transformation(sixdof):
    x,y,z, r,p,y = sixdof
    vec = np.array([x,y,z, r * np.pi/180, p * np.pi/180, y * np.pi / 180])
    rot = np.matmul(np.matmul(Rz(vec[5]), Ry(vec[4]) ), Rx(vec[3]) )
    T   = np.identity(4)
    T[:3,:3] = rot[:3, :3]
    T[:3, 3] = np.transpose(vec[:3])
    return T

# The one that we measured for realsense2lidar
#c2l_vec = np.array([0.19,  0.04, 0.2, -0.0124 * np.pi / 180, -89.5533* np.pi/180, 91.2846*np.pi/180])
# The one we actually used in this projection package: I tuned the numbers based on 
# my eyes' judgement of projection quality!

if __name__ == "__main__":

    x_lb3_c = np.genfromtxt(sys.argv[1], delimiter=',')
    
    x_body_lb3 = np.genfromtxt ("nclt_cams/x_body_lb3.csv", delimiter=",")



    T_lb3_c = sixdof_to_transformation(x_lb3_c )
    T_body_lb3 = sixdof_to_transformation(x_body_lb3)

    T_lb3_body = np.linalg.inv(T_body_lb3)
    T_c_lb3 = np.linalg.inv(T_lb3_c)
    # because nclt lidar are in the body frame .........
    T_c_body = np.matmul(T_c_lb3, T_lb3_body)
    np.save(sys.argv[2], (T_c_body))
    
