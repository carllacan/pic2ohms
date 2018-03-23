#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 11:02:32 2018

@author: carles
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:56:17 2018

@author: carles
"""

import numpy as np
import random

import utils
from goniometer import Goniometer

# TODO: use a generator

dataset = 5
pics_loc = 'datasets/dataset{0}/dataset{0}_imgs.csv'.format(dataset)
angs_loc = 'datasets/dataset{0}/dataset{0}_angles.csv'.format(dataset)

pics = np.genfromtxt(pics_loc, delimiter=',')
angs = np.genfromtxt(angs_loc, delimiter=',', dtype=int)

 # Create the NN model
 
goniometer = Goniometer(input_shape=(48,48), 
                        hidden_layers=(42,30,10))

epochs = 10
batch_size = 20
goniometer.train(pics, angs, epochs, batch_size)

# test the goniometer with a number of resistors

test_pics =  []
for p in range(25):
    ind = random.randint(0, pics.shape[0] - 1)
    test_pics.append(pics[ind,:].reshape(48,48))
    
utils.test_goniometer(goniometer, test_pics)

# next, a hard test, just for fun

#resistors = Image.open('resistors.png', mode='r')
#resistors = resistors.convert(mode='F')
 
#utils.test_pic(goniometer, np.asarray(resistors), show_probs=False)

goniometer.save('datasets/dataset{0}/best_model'.format(dataset))

