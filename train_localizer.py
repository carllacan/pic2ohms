#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import random

from PIL import Image

import utils
from localizer import Localizer
from picture_generator import PictureGenerator

from keras.utils.np_utils import to_categorical

# Create the NN model
 
localizer = Localizer(input_shape=(48, 48), 
                      hidden_layers=(42,30,10))

# Create the data generator

generator = PictureGenerator(batch_size = 15, 
                             batches_per_epoch = 500,
                             return_angles = False,
                             resistor_prob = 0.5,
                             real_backgrounds = True,
                             angle_num = 8,
                             flatten=False)
          
# Train the model
    
epochs = 10

localizer.train(generator, epochs)

#i = random.randint(0, pics.shape[0] - 1)
#pic = pics[i,:].reshape(240,240)
#utils.test_localizer(localizer, pic, show_probs=False)

#resistors = Image.open('test_pictures/hard_test.png', mode='r')
#resistors = resistors.convert(mode='F')
# 
#utils.test_localizer(localizer, np.asarray(resistors), show_probs=False)

#localizer.save('datasets/dataset{0}/best_model'.format(dataset))













