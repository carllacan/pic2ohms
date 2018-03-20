#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

from PIL import Image

import utils
from reader import Reader
from resistor_generator import ResistorGenerator

from keras.utils.np_utils import to_categorical

# Create the NN model
 
localizer = Reader(input_shape=(48, 48), 
                      hidden_layers=(42,30,10))

# Create the data generator

generator = ResistorGenerator(batch_size = 15, 
                             batches_per_epoch = 500)
          
# Train the model
    
epochs = 10

localizer.train(generator, epochs)