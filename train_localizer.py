#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:56:17 2018

@author: carles
"""

import numpy as np
import random

from PIL import Image

import utils
from localizer import Localizer

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
 
localizer = Localizer(input_shape=(240, 240), 
                      hidden_layers=(42,30,10))

epochs = 20
batch_size = 20
localizer.train(pics, boxs, epochs, batch_size)

i = random.randint(0, pics.shape[0] - 1)
pic = pics[i,:].reshape(240,240)
utils.test_localizer(localizer, pic, show_probs=True)
#print(localizer.predict(pic))

# a hard test, just for fun

resistors = Image.open('resistors.png', mode='r')
resistors = resistors.convert(mode='F')
 
utils.test_localizer(localizer, np.asarray(resistors), show_probs=True)

localizer.save('datasets/dataset{0}/best_model'.format(dataset))













