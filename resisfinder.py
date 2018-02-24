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
    img = img.reshape(240,240)
    fig,ax = plt.subplots(1)
    plt.imshow(img, cmap = 'gray' )
    for b in boxes:
        x, y = b[0], b[1]
        w, h = b[2] - b[0], b[3] - b[1]
        rect = patches.Rectangle((x,y), w, h, linewidth=1,edgecolor='r',facecolor='none')

        ax.add_patch(rect)
        
def test_pic(model, img, box):
    """ Predicts the position of resistors on an image"""
    sec_dim = 48 # section size
    stride = 48 # section stride
    
    maxi = int((picshape[1] - sec_dim) / stride) +1
    maxj = int((picshape[0] - sec_dim) / stride) +1
    boxes = []
    for i in range(maxi):
        for j in range(maxj):
            # Section box
            x1, y1 = i*sec_dim, j*sec_dim
            x2, y2 = x1 + sec_dim, y1 + sec_dim
            box = np.array((x1, y1, x2, y2))
            sec = img.reshape(picshape)[y1:y2, x1:x2].flatten() 
            if model.predict_classes(np.array([sec]))[0]:
                boxes.append(box)
    
    show_with_boxes(img, boxes)

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
secs = []
labels = []

sec_dim = 48 # section size
stride = 48 # section stride

maxi = np.floor((picshape[1] - sec_dim) / stride) + 1
maxj = np.floor((picshape[0] - sec_dim) / stride) + 1

for n in range(data_size):
    # Get a random image index
    img = random.randint(0, imgs.shape[0]-1)
    # Indexes of the random section
    i = random.randint(0, maxi-1)
    j = random.randint(0, maxj-1)
    # Section box
    x1, y1 = i*sec_dim, j*sec_dim
    x2, y2 = x1 + sec_dim, y1 + sec_dim
    box = np.array((x1, y1, x2, y2))
    label = 0
    for b in boxs[img,:]:
        # If the bsection exactly contains a resistor mark as 1
        # I may need to relax this definition of error later on
        # Perhaps use the overlapping area with actual resistors
        if (box == b).all():
            label = 1
    # It would be faster if I didn't reshape
    sec = imgs[img,:].reshape(picshape)[y1:y2, x1:x2].flatten()
    secs.append(sec)
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
model.fit(np.array(secs[0:ntrain]), 
          np.array(labels[0:ntrain]), 
          epochs=epochs, batch_size=bsize)

# since sections are already created at random I might not need to
#   select the training samples randomly

### Validation ####

nval = int(data_size*0.2)
#preds = model.predict_classes(np.array(secs[0:nval]), 
#                              verbose=False)
#acc = (preds.transpose()==np.array(labels[0:nval])).sum()/nval
a, acc = model.evaluate(np.array(secs[0:nval]), np.array(labels[0:nval]),
                        batch_size=128)
print("Accuracy: {:4.2f} %".format(acc*100))


i = random.randint(0, imgs.shape[0] - 1)
test_pic(model, imgs[i,:], boxs[i,:])





    
    
# TODO: put the model on a separate module, so that I can call
#   a different one each time.
#   Include an argument at creation to decide the size of the input
#   Include train and test functions.















