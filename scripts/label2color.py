#!/usr/bin/python

# source: https://github.com/UMich-BipedLab/segmentation_projection
# maintainer: Ray Zhang    rzh@umich.edu


background = 255

# RGB
label_to_color = {                                                                                                              
    2: (250,250,250), # road
    #3: (128, 64,128), # sidewalk
    3: (250, 250, 250), # sidewalk
    5: (250,128,0), # building
    10: (192,192,192), # pole
    12: (250,250,0 ), # traffic sign
    6: (107,142, 35), # vegetation
    4: (128,128, 0 ), # terrain
    13: ( 135, 206, 235 ),  # sky
    1: (  30, 144, 250 ),  # water
    8 :(220, 20, 60), # person
    7: ( 0, 0,142),  # car
    9 : (119, 11, 32),# bike
    11 : (123, 104, 238), # stair

    0: (0 ,0, 0),       # background
    background: (0,0,0) # background
}

label_to_color = {                                                                                                              
    2: (245,190,133), # road
    #3: (128, 64,128), # sidewalk
    3: (234, 190, 133), # sidewalk
    5: (93,43,77), # building
    10: (122,21,14), # pole
    12: (0,0,0 ), # traffic sign
    6: (211,175, 141), # vegetation
    4: (26,14, 24 ), # terrain
    13: ( 140, 60, 78 ),  # sky
    1: (  0, 0, 0 ),  # water
    8 :(56, 7, 8), # person
    7: ( 0, 0,0),  # car
    9 : (0, 0, 0),# bike
    11 : (0, 0, 0), # stair

    0: (0 ,0, 0),       # background
    background: (0,0,0) # background
}
