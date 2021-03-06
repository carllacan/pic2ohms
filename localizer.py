#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:08:57 2018

@author: carles
"""

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

class Localizer():
    
    """ ANN model that localizes resistors in a picture """
    def __init__(self, **params):
        self.input_shape = (48,48)
        
        if 'filepath' in params.keys():
            self.load_model(params['filepath'])
        else:
            self.create_model(**params)
            
    def create_model(self, **params):
        
        hidden_layers = params['hidden_layers']
        rel = lambda: l2(0.01)
        self.input_shape = (48,48) 
        self.model = Sequential()
        input_dim = self.input_shape[0]*self.input_shape[1]
        self.model.add(Dense(hidden_layers[0], 
                             input_dim = input_dim, 
                             kernel_initializer="random_normal", 
                             activation="hard_sigmoid", 
                             kernel_regularizer=rel()))
        for l in hidden_layers[1:]:
            self.model.add(Dense(l, 
                                 kernel_initializer="random_normal", 
                                 activation="hard_sigmoid", 
                                 kernel_regularizer=rel()))
        self.model.add(Dense(2, 
                             kernel_initializer="random_normal", 
                             activation="softmax",
                             kernel_regularizer=rel()))
                        
        # for probs log-loss seeems t be better
        self.model.compile(loss="categorical_crossentropy", 
                           optimizer="adam", metrics=['accuracy'])
        
    def create_model(self, **params):
        # Convolutional model
        
        hidden_layers = params['hidden_layers']
        rel = lambda: l2(0.01)
        self.model = Sequential()
        input_dim = self.input_shape[0]*self.input_shape[1]
        self.model.add(Conv2D(30, (4, 4), 
                              strides = (2,2),
                              padding="valid",
                              input_shape=(48, 48, 1),
                              data_format="channels_last",
                              kernel_initializer="random_normal",
                              activation='relu',
                              kernel_regularizer=rel()))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Conv2D(22, (4, 4), 
                              strides = (2,2),
                              padding="valid",
                              data_format="channels_last",
                              kernel_initializer="random_normal",
                              activation='relu',
                              kernel_regularizer=rel()))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(30, 
                             kernel_initializer="random_normal", 
                             activation="softmax",
                             kernel_regularizer=rel()))
        self.model.add(Dense(2, 
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
        
        a, acc = self.model.evaluate_generator(generator, steps=1000)
        print("Accuracy: {:4.2f} %".format(acc*100)) 
        
    def predict(self, pic):
        """ Takes a picture and returns an array the boxes considered,
        the predictions for each box and trhe probabilities. """
 
        secs = []
        sec_dim = self.input_shape[0]
        boxes = utils.make_sections(pic.shape, sec_dim, pic.shape[0]//4)
        for b in boxes:
            x1, y1 = b[0], b[1]
            x2, y2 = b[2], b[3]
            secs.append(pic[y1:y2, x1:x2].reshape(48, 48, 1 ))
        preds = self.model.predict_classes(np.array(secs))
        probs = self.model.predict(np.array(secs))

        return boxes, preds, probs
    
    def localize(self, pic):
        """ Takes a picture and returns an array of resistor boxes"""
  
        boxes, preds, probs = self.predict(pic)
        found = []
        for i in range(len(boxes)):
            if preds[i]:
                found.append(boxes[i])
                
        return found
    