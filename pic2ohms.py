#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:20:33 2018

@author: carles
"""

# This script uses the localizer, goniometer and reader modules to
# find, rotate and read resistors in an image.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate as scprot

import random
from PIL import Image

import utils

from localizer import Localizer
from goniometer import Goniometer

# TODO: showcase localizer and goniometer
# create a picture
# load the best localizer and goniometer
# load an image 
# make a b/w version
# use  localizer to find resistors
# pass the subimages to goniometer to read the angles
# get the subimages from the color picture
# rotate the subimages back to 0º and output them.

# TODO: standarize what kind of arguments each object takes, and be coherent
# TODO: test_localizer converts to BW by itself
# TODO: goniometer.get_angle()


localizer = Localizer(filepath='datasets/dataset4/best_model')
goniometer = Goniometer(filepath='datasets/dataset5/best_model')

picture_bw = picture.convert('F')

resis_boxs = localizer.localize(np.asarray(picture_bw))
plt.show()
resis_pics = [] # cropped pictures of the resistors

for b in resis_boxs:
    x1, y1 = b[0], b[1]
    x2, y2 = b[2], b[3]
    angle_ind = goniometer.predict(np.asarray(picture_bw)[y1:y2,x1:x2])
    angle = goniometer.angle_list[angle_ind[0]]
    sec = picture.crop((x1, y1, x2, y2))
    print(angle)
    plt.figure(figsize=(1,1))
    plt.axis("off")
    plt.imshow(np.asarray(sec),cmap = 'gray' )
    plt.show()
#    sec = scprot(np.asarray(sec), -angle)
    sec = sec.rotate(-angle)
    plt.figure(figsize=(1,1))
    plt.axis("off")
    plt.imshow(np.asarray(sec),cmap = 'gray' )
    plt.show()
    resis_pics.append(sec)













# TODO: implement the reader and use it to tag the subimages