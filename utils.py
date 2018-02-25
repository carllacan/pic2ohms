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

def show_with_boxes(img, boxes, preds, probs = []):
    """ Show an image with a series of boxes drawn over it"""
    plt.figure()
    fig,ax = plt.subplots(1)
    plt.imshow(img, cmap = 'gray' )
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
        if probs != []:
            color = 'b' if preds[i] else 'r'
            x, y = int(x+w/2), int(y+h/2)
            r = int(0.5*max(probs[i])*w/4)
            rect = patches.Circle((x,y), r, 
                                  alpha=0.75,
                                  linewidth=0,
                                  edgecolor='k',
                                  facecolor=color)
            ax.add_patch(rect)
        
def show_with_angles(pic, angle, probs=[]):
    plt.figure()
    fig,ax = plt.subplots(1)
    plt.imshow(pic, cmap = 'gray' )
    angle_num = len(probs)
    for i in range(angle_num):
#        line = patches.ConnectionPatch((0,0), (0,0))
        plt.plot((0,0), (0,0))
        
    
        
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


def test_pic(localizer, pic, show_probs=False):
     # TODO: change to test loc
    """ Predicts the position of resistors on an picture and shows it"""
#    sec_dim = 48 # section size
#    stride = 48 # section stride
#    
#    boxes = make_sections(pic, sec_dim, stride)
#    found = []
#    for b in boxes:
#        x1, y1 = b[0], b[1]
#        x2, y2 = b[2], b[3]
#        sec = pic[y1:y2, x1:x2].flatten() 
#        pred = model.predict_classes(np.array([sec]))[0]
#        if pred == 1:
#            found.append(b)
    boxes, preds, probs = localizer.predict(pic) 
    if show_probs:           
        show_with_boxes(pic, boxes, preds, probs)
    else:         
        show_with_boxes(pic, boxes, preds)
        
def test_gon(goniometer, pic, show_probs=False):
    
     angle, probs = goniometer.predict(pic)
     
     show_with_angles(pic, angle, probs)
        