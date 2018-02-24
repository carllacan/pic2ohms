#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:11:36 2018

@author: carles
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from keras.utils.np_utils import to_categorical

def show_with_boxes(img, boxes):
    """ Show an image with a series of boxes drawn over it"""
    plt.figure()
    fig,ax = plt.subplots(1)
    plt.imshow(img, cmap = 'gray' )
    for b in boxes:
        x, y = b[0], b[1]
        w, h = b[2] - b[0], b[3] - b[1]
        rect = patches.Rectangle((x,y), w, h, 
                                 linewidth=1,
                                 edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
        
def make_sections(pic, sec_dim, stride):
    """ Takes a picture and returns boxes of sections """
    maxi = int((pic.shape[1] - sec_dim) / stride) +1
    maxj = int((pic.shape[0] - sec_dim) / stride) +1
    boxes = []
    for i in range(maxi):
        for j in range(maxj):
            # Section box
            x1, y1 = i*stride, j*stride
            x2, y2 = x1 + sec_dim, y1 + sec_dim
            box = np.array((x1, y1, x2, y2))
            boxes.append(box)
    return boxes


def test_pic(model, pic):
    """ Predicts the position of resistors on an picture"""
    sec_dim = 48 # section size
    stride = 48 # section stride
    
    boxes = make_sections(pic, sec_dim, stride)
    found = []
    for b in boxes:
        x1, y1 = b[0], b[1]
        x2, y2 = b[2], b[3]
        sec = pic[y1:y2, x1:x2].flatten() 
        pred = model.predict_classes(np.array([sec]))[0]
        if pred == 1:
            found.append(b)
    show_with_boxes(pic, found)