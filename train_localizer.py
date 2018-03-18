#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import random

from PIL import Image

import utils
from localizer import Localizer

from keras.utils.np_utils import to_categorical

dataset = 4
imgs_loc = 'datasets/dataset{0}/dataset{0}_imgs.csv'.format(dataset)
boxs_loc = 'datasets/dataset{0}/dataset{0}_boxs.csv'.format(dataset)

pics = np.genfromtxt(imgs_loc, delimiter=',')
boxs = np.genfromtxt(boxs_loc, delimiter=',')

# Resize the box lists
boxs = boxs.reshape(boxs.shape[0], -1, 4)

# dimensions of the whole pictures, constant for now
picshape = 240, 240 # height and width

# Create the NN model
 
localizer = Localizer(input_shape=(48, 48), 
                      hidden_layers=(42,30,10))

# Make the training data
picn = pics.shape[0] # number of pics

sec_dim = localizer.input_shape[0] # section size
stride = int(sec_dim/1) # section stride
secs = utils.make_sections(picshape, sec_dim, stride)
    
data_size = picn*25 # sections to create from each picture
data = []
labels = []

for n in range(data_size):
    # Get a random image index
    pic_ind = random.randint(0, picn-1)
    pic = pics[pic_ind,:].reshape(picshape)

    sec = random.choice(secs) # choosen section
    x1, y1 = sec[0], sec[1]
    x2, y2 = sec[2], sec[3]
    cropped = pic.reshape(picshape)[y1:y2, x1:x2].flatten()
    data.append(cropped)
    
    label = 0
    for b in boxs[pic_ind,:]:
        # If the bsection exactly contains a resistor mark as 1
        # I may need to relax this definition of error later on
        # Perhaps use the overlapping area with actual resistors
        if (sec == b).all():
            label = 1
#    labels = [to_categorical(1 if (sec == b).all() else 0, 2) for b in boxs[pic_ind,:]]
            

    labels.append(to_categorical(label, 2))
           
# Train the model
    
epochs = 20
batch_size = 20
localizer.train(data, labels, epochs, batch_size)

i = random.randint(0, pics.shape[0] - 1)
pic = pics[i,:].reshape(240,240)
utils.test_localizer(localizer, pic, show_probs=True)
#print(localizer.predict(pic))

# a hard test, just for fun

resistors = Image.open('test_pictures/resistors.png', mode='r')
resistors = resistors.convert(mode='F')
 
utils.test_localizer(localizer, np.asarray(resistors), show_probs=False)

localizer.save('datasets/dataset{0}/best_model'.format(dataset))













