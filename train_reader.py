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
 
localizer = Reader(input_shape=(16, 48, 3))

# Create the data generator

generator = ResistorGenerator(batch_size = 15, 
                             batches_per_epoch = 200,
                             labeled_band=2)
          
# Train the model
    
epochs = 5

localizer.train(generator, epochs)