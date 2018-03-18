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

def test_localizer(localizer, pic, show_probs = False):
    """ Show an image with a series of boxes drawn over it"""
    
    boxes, preds, probs = localizer.predict(pic) 
        
    fig,ax = plt.subplots(1)
    plt.imshow(pic, cmap = 'gray' )
    plt.axis("off")
    for i in range(len(boxes)):
        box = boxes[i]
        x, y = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        if preds[i]:
            rect = patches.Rectangle((x,y), w, h, 
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
        if show_probs:
            color = 'b' if preds[i] else 'r'
            x, y = int(x+w/2), int(y+h/2)
            r = int(0.5*max(probs[i])*w/4)
            rect = patches.Circle((x,y), r, 
                                  alpha=0.75,
                                  linewidth=0,
                                  edgecolor='k',
                                  facecolor=color)
            ax.add_patch(rect)
        
def test_goniometer(goniometer, test_pics):
    """ Visuzlie the results of a goniometer with 25 resistor pics"""
    w, h = goniometer.input_shape
    fig, axs = plt.subplots(5, 5)
    fig.subplots_adjust(wspace = 0.1, hspace=0.1)
    angle_list = goniometer.angle_list
    for i in range(5):
        for j in range(5):
            pic = test_pics[i+j*5]
            pred, probs = goniometer.predict(pic)
            probs = probs/probs[pred]
            ax = axs[i][j]
            ax.imshow(pic, cmap = 'gray' )
            ax.axis("off")
            for a in range(len(angle_list)):
                r = (w/2-1)*probs[a]
                x = w/2+r*np.cos(-angle_list[a]*np.pi/180)
                y = h/2+r*np.sin(-angle_list[a]*np.pi/180)
                if angle_list[a] == angle_list[pred]:
                    color = 'b'
                else:
                    color = 'r'
                ax.plot([int(w/2), x], [int(h/2), y], color=color)
        
    
        
def make_sections(pic_shape, sec_dim, stride):
    """ Takes a picture and returns boxes of sections """
    maxi = int((pic_shape[1] - sec_dim) / stride) +1
    maxj = int((pic_shape[0] - sec_dim) / stride) +1
    boxes = []
    for i in range(maxi):
        for j in range(maxj):
            # Section box
            x1, y1 = i*stride, j*stride
            x2, y2 = x1 + sec_dim, y1 + sec_dim
            box = np.array((x1, y1, x2, y2))
            boxes.append(box)
    return boxes



        