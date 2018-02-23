#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:56:17 2018

@author: carles
"""

# As a first step, the simplest possible case:
    
# A neural network that takes a 48x48 picture and decides whether
# there is a resistor or not. Background is uniform.

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.regularizers import l2, l1
#from keras.utils.np_utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def show_with_boxes(img, boxes):
    """ Show an image with a series of boxes drawn over it"""
    plt.figure()
    img = img.reshape(240,240)
    fig,ax = plt.subplots(1)
    plt.imshow(img, cmap = 'gray' )
    for b in boxes:
        x, y = b[0], b[1]
        w, h = b[2] - b[0], b[3] - b[1]
        rect = patches.Rectangle((x,y), w, h, linewidth=1,edgecolor='r',facecolor='none')

        ax.add_patch(rect)
        
imgs = np.genfromtxt('dataset1/dataset1_imgs.csv', delimiter=',')
boxs = np.genfromtxt('dataset1/dataset1_boxs.csv', delimiter=',')

# Resize the images?

# Resize the box lists
boxs = boxs.reshape(5, -1, 4)

for i in range(imgs.shape[0]):
    boxes = boxs[i,:]
    show_with_boxes(imgs[i,:], boxes)