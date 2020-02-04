#!/usr/bin/python

# source: https://github.com/UMich-BipedLab/segmentation_projection
# maintainer: Lu Gan ganlu@umich.edu

label_to_traversability = {
    1: 0, # water
    2: 1, # road
    3: 1, # sidewalk
    4: 1, # terrain
    5: 0, # building
    6: 0, # vegetation
    7: 0, # car
    8: 0, # person
    9: 0, # bike
    10: 0, # pole
    11: 1, # stair
    12: 0, # traffic sign
    13: 0, # sky
    0: 0 # background
}
