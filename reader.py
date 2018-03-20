#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.regularizers import l2, l1
from keras.utils.np_utils import to_categorical
from keras.models import load_model

import numpy as np
import random
import utils
#import h5py 

class Reader():
    
    """ ANN model that localizes resistors in a picture """
    def __init__(self, **params):
        self.input_shape = params['input_shape']
        
        if 'filepath' in params.keys():
            self.load_model(params['filepath'])
        else:
            self.create_model(**params)
            
        
    def create_model(self, **params):
        # Convolutional model
        
        rel = lambda: l2(0.2)
        self.model = Sequential()
        self.model.add(Conv2D(24, (3, 3), 
                              strides = (1,1),
                              padding="valid",
                              input_shape=self.input_shape,
                              data_format="channels_last",
                              kernel_initializer="random_normal",
                              activation='relu',
                              kernel_regularizer=rel()))
        self.model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
        self.model.add(Conv2D(16, (4, 4), 
                              strides = (2,2),
                              padding="valid",
                              data_format="channels_last",
                              kernel_initializer="random_normal",
                              activation='relu',
                              kernel_regularizer=rel()))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(100, 
                             kernel_initializer="random_normal", 
                             activation="relu",
                             kernel_regularizer=rel()))
        self.model.add(Dense(60, 
                             kernel_initializer="random_normal", 
                             activation="relu",
                             kernel_regularizer=rel()))
        self.model.add(Dense(30, 
                             kernel_initializer="random_normal", 
                             activation="relu",
                             kernel_regularizer=rel()))
        self.model.add(Dense(4, 
                             kernel_initializer="random_normal", 
                             activation="softmax",
                             kernel_regularizer=rel()))
                        
        # for probs log-loss seeems t be better
        self.model.compile(loss="categorical_crossentropy", 
                           optimizer="adam", metrics=['accuracy'])
        
    def load_model(self, filepath):
        """Loads a model from a file """
        self.model = load_model(filepath)
    
    def save(self, filepath):
        """Saves  model to a file """
        self.model.save(filepath)
        
         
    def train(self, generator, epochs):
        """ Trains the model with a list of pictures """
        # generator version

        #### Training ####

        self.model.fit_generator(generator, epochs=epochs)
                
        ### Validation ####
        
        a, acc = self.model.evaluate_generator(generator, steps=100)
        print("Accuracy: {:4.2f} %".format(acc*100)) 
        

    