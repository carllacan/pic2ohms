#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:56:17 2018

@author: carles
"""

# As a first step, the simplest possible case:

# A neural network that takes a 48x48 picture and decides whether
# there is a resistor or not. Background is uniform.

#
#import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import numpy as np
import random


import utils
#from localizers import Localizer1
import localizers

dataset = 1
imgs_loc = 'datasets/dataset{0}/dataset{0}_imgs.csv'.format(dataset)
boxs_loc = 'datasets/dataset{0}/dataset{0}_boxs.csv'.format(dataset)

pics = np.genfromtxt(imgs_loc, delimiter=',')
boxs = np.genfromtxt(boxs_loc, delimiter=',')

# Resize the box lists
boxs = boxs.reshape(boxs.shape[0], -1, 4)

# dimensions of the whole pictures, constant for now
picshape = 240, 240 # height and width

    
 # Create the NN model
 
localizer = localizers.Localizer1(picshape)

localizer.train(pics, boxs)

i = random.randint(0, pics.shape[0] - 1)
utils.test_pic(localizer.model, pics[i,:].reshape(240,240))


    















