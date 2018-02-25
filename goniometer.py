#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 23:24:39 2018

@author: carles
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2, l1
from keras.utils.np_utils import to_categorical

import numpy as np
import random
import utils
import h5py 

class Goniometer():
    
    """ ANN model that gets the angle of a resistor """
    def __init__(self, **params):

        self.input_shape = params['input_shape']
        input_dim = self.input_shape[0]*self.input_shape[1]
        hidden_layers = params['hidden_layers']
        rel = lambda: l2(0.2)

        self.model = Sequential()
        for l in hidden_layers:
            self.model.add(Dense(l, 
                                 input_dim = input_dim,
                                 kernel_initializer="uniform", 
                                 activation="relu", 
                                 kernel_regularizer=rel()))
        self.model.add(Dense(8, 
                             kernel_initializer="uniform", 
                             activation="softmax",
                             kernel_regularizer=rel()))
                        
        self.model.compile(loss="categorical_crossentropy", 
                           optimizer="adam", metrics=['accuracy'])
        
    def train(self, pics, angs, epochs, batch_size):
        
        # Make the training data
        data_size = pics.shape[0] # number of pics
        
        data = np.array(pics)
        
        self.angle_list = list(range(0, 360, 45))
        labels = []
        for a in angs:
            c = self.angle_list.index(a)
            labels.append(c)
        labels = to_categorical(labels)
        
        #### Training ####
        
        ntrain = int(data_size*0.8)
        train_inds = random.sample(range(data_size), ntrain)
    
        self.model.fit(data[train_inds,:], 
                       labels[train_inds,:], 
                       batch_size, epochs)
               
        ### Validation ####
        
        nval = int(data_size*0.2)
        val_inds = random.sample(range(data_size), nval)
        a, acc = self.model.evaluate(data[val_inds], 
                                     labels[val_inds],
                                     batch_size=128)
        print("Accuracy: {:4.2f} %".format(acc*100))

    
    def predict(self, pic):
        """ Takes a picture and returns an array of resistor boxes"""
        
        angle = self.model.predict_classes(np.array([pic.flatten()]))[0]
        probs = self.model.predict(np.array([pic.flatten()]))[0]
        return angle, probs
    
   
    def save(self, f):
        """Saves current parameters of the model to a file """
        self.model.save(f)