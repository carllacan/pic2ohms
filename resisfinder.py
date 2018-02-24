#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:56:17 2018

@author: carles
"""

# As a first step, the simplest possible case:

# A neural network that takes a 48x48 picture and decides whether
# there is a resistor or not. Background is uniform.

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2, l1
from keras.utils.np_utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import random


def show_with_boxes(img, boxes):
    """ Show an image with a series of boxes drawn over it"""
    plt.figure()
    fig,ax = plt.subplots(1)
    plt.imshow(img, cmap = 'gray' )
    for b in boxes:
        x, y = b[0], b[1]
        w, h = b[2] - b[0], b[3] - b[1]
        rect = patches.Rectangle((x,y), w, h, linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        
def make_sections(pic, sec_dim, stride):
    """ Takes a picture and returns boxes of sections """
    maxi = int((pic.shape[1] - sec_dim) / stride) +1
    maxj = int((pic.shape[0] - sec_dim) / stride) +1
    boxes = []
    for i in range(maxi):
        for j in range(maxj):
            # Section box
            x1, y1 = i*sec_dim, j*sec_dim
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
        if model.predict_classes(np.array([sec]))[0]:
            found.append(b)
    
    show_with_boxes(pic, found)

imgs = np.genfromtxt('dataset1/dataset1_imgs.csv', delimiter=',')
boxs = np.genfromtxt('dataset1/dataset1_boxs.csv', delimiter=',')

# Resize the images?

# Resize the box lists
boxs = boxs.reshape(5, -1, 4)

#for i in range(imgs.shape[0]):
#    boxes = boxs[i,:]
#    show_with_boxes(imgs[i,:], boxes)


# dimensions of the whole pictures, constant for now
picshape = 240, 240 # height and width



# Create the random sections (the training data)

data_size = 8000 # sections to create
data = []
labels = []

sec_dim = 48 # section size
stride = 48 # section stride

maxi = np.floor((picshape[1] - sec_dim) / stride) + 1
maxj = np.floor((picshape[0] - sec_dim) / stride) + 1

for n in range(data_size):
    # Get a random image index
    img_ind = random.randint(0, imgs.shape[0]-1)
    img = imgs[img_ind,:].reshape(picshape)

    secs = make_sections(img.reshape(picshape), sec_dim, stride)
    sec = random.choice(secs) # choosen section
    label = 0
    for b in boxs[img_ind,:]:
        # If the bsection exactly contains a resistor mark as 1
        # I may need to relax this definition of error later on
        # Perhaps use the overlapping area with actual resistors
        if (sec == b).all():
            label = 1
            
    x1, y1 = sec[0], sec[1]
    x2, y2 = sec[2], sec[3]
    cropped = img.reshape(picshape)[y1:y2, x1:x2].flatten()
    data.append(cropped)
    labels.append(to_categorical(label, 2))
    
 # Create the NN model
 
hidden_neurons = 12
rel = lambda: l2(0.005)

model = Sequential()
model.add(Dense(hidden_neurons, 
                input_dim = sec_dim*sec_dim, 
                kernel_initializer="uniform", 
                activation="relu", 
                kernel_regularizer=rel()))
model.add(Dense(int(hidden_neurons/2), 
                input_dim = sec_dim*sec_dim, 
                kernel_initializer="uniform", 
                activation="relu", 
                kernel_regularizer=rel()))
model.add(Dense(2, 
                kernel_initializer="uniform", 
                activation="softmax",
                kernel_regularizer=rel()))
                
model.compile(loss="categorical_crossentropy", 
              optimizer="adam", metrics=['accuracy'])

#### Training ####
ntrain = int(data_size*0.8)
epochs = 3
bsize = 20
model.fit(np.array(data[0:ntrain]), 
          np.array(labels[0:ntrain]), 
          epochs=epochs, batch_size=bsize)

# since sections are already created at random I might not need to
#   select the training samples randomly

### Validation ####

nval = int(data_size*0.2)
#preds = model.predict_classes(np.array(secs[0:nval]), 
#                              verbose=False)
#acc = (preds.transpose()==np.array(labels[0:nval])).sum()/nval
a, acc = model.evaluate(np.array(data[0:nval]), np.array(labels[0:nval]),
                        batch_size=128)
print("Accuracy: {:4.2f} %".format(acc*100))


i = random.randint(0, imgs.shape[0] - 1)
test_pic(model, imgs[i,:].reshape(240,240))


    
# TODO: put the model on a separate module, so that I can call
#   a different one each time.
#   Include an argument at creation to decide the size of the input
#   Include train and test functions.















