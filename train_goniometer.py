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

from PIL import Image

import utils
from goniometer import Goniometer

dataset = 5
imgs_loc = 'datasets/dataset{0}/dataset{0}_imgs.csv'.format(dataset)
angs_loc = 'datasets/dataset{0}/dataset{0}_angles.csv'.format(dataset)

pics = np.genfromtxt(imgs_loc, delimiter=',')
angs = np.genfromtxt(angs_loc, delimiter=',', dtype=int)

 # Create the NN model
 
goniometer = Goniometer(picdim=48, 
                        hidden_layers=(42,30,10))

epochs = 20
batch_size = 30
goniometer.train(pics, angs, epochs, batch_size)

i = random.randint(0, pics.shape[0] - 1)
pic = pics[i,:].reshape(48,48)
utils.test_gon(goniometer, pic, show_probs=True)

# a hard test, just for fun

#resistors = Image.open('resistors.png', mode='r')
#resistors = resistors.convert(mode='F')
 
#utils.test_pic(goniometer, np.asarray(resistors), show_probs=False)

goniometer.save('datasets/dataset{0}/best_model'.format(dataset))

