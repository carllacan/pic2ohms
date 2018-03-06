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
from keras.models import load_model

import numpy as np
import random
import utils
import h5py 

class Goniometer():
    
    """ ANN model that gets the angle of a resistor """
    def __init__(self, **params):
        # TODO: save and load the angle list
        self.angle_list = list(range(0, 360, 45))
        if 'filepath' in params.keys():
            self.load_model(params['filepath'])
        else:
            self.create_model(**params)
            
    def create_model(self, **params):
        self.input_shape = (48, 48)#params['input_shape']
        hidden_layers = params['hidden_layers']
        rel = lambda: l2(0.2)

        self.model = Sequential()
        input_dim = self.input_shape[0]*self.input_shape[1]
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
                
    def load_model(self, filepath):
        """Loads a model from a file """
        self.model = load_model(filepath)
    
    def save(self, filepath):
        """Saves  model to a file """
        self.model.save(filepath)
        
    def train(self, pics, angs, epochs, batch_size):
        """ Trains the model with a list of pictures """
        
        # Make the training data
        data_size = pics.shape[0] # number of pics
        
        data = np.array(pics)
        
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
        """ Takes a resistor picture and returns the angle and the 
        probabilities for each angle"""
        
        angle = self.model.predict_classes(np.array([pic.flatten()]))[0]
        probs = self.model.predict(np.array([pic.flatten()]))[0]
        return angle, probs
    
