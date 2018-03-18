#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:08:57 2018

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
#import h5py 

class Localizer():
    
    """ ANN model that localizes resistors in a picture """
    def __init__(self, **params):
        self.input_shape = (48,48) #params['input_shape']
        self.stride = self.input_shape[0]//2#int(self.sec_dim/3)
        # TODO: make it store the input_shape, or get it from the arch
        # TODO also the stride
        if 'filepath' in params.keys():
            self.load_model(params['filepath'])
        else:
            self.create_model(**params)
            
    def create_model(self, **params):
        
        hidden_layers = params['hidden_layers']
        rel = lambda: l2(0.2)
        self.input_shape = (48,48) #params['input_shape']
        self.stride = self.input_shape[0]//2#int(self.sec_dim/3)
        self.model = Sequential()
        input_dim = self.input_shape[0]*self.input_shape[1]
        for l in hidden_layers:
            self.model.add(Dense(l, 
                                 input_dim = input_dim, 
                                 kernel_initializer="uniform", 
                                 activation="relu", 
                                 kernel_regularizer=rel()))
        self.model.add(Dense(2, 
                             kernel_initializer="uniform", 
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
        
    def train(self, data, labels, epochs, batch_size):
        """ Trains the model with a list of pictures """
#        
#        picn = pics.shape[0] # number of pics
#        data_size = picn*25 # sections to create from each picture
#        data = []
#        labels = []
#        
#        sec_dim = self.input_shape[0] # section size
#        stride = self.stride # section stride
#        
#        ### Training data creation ###
#        
#        for n in range(data_size):
#            # Get a random image index
#            pic_ind = random.randint(0, picn-1)
#            pic = pics[pic_ind,:].reshape(self.picshape)
#        
#            secs = utils.make_sections(pic.reshape(self.picshape), 
#                                       sec_dim, stride)
#            sec = random.choice(secs) # choosen section
#            label = 0
#            for b in boxs[pic_ind,:]:
#                # If the bsection exactly contains a resistor mark as 1
#                # I may need to relax this definition of error later on
#                # Perhaps use the overlapping area with actual resistors
#                if (sec == b).all():
#                    label = 1
#                    
#            x1, y1 = sec[0], sec[1]
#            x2, y2 = sec[2], sec[3]
#            cropped = pic.reshape(self.picshape)[y1:y2, x1:x2].flatten()
#            data.append(cropped)
#            labels.append(to_categorical(label, 2))
           
        #### Training ####
        
        ntrain = int(len(data)*0.8)
        # The images are already created at random, I probably don't need to
        #   select the training datapoints randomly
        
        self.model.fit(np.array(data[0:ntrain]), 
                       np.array(labels[0:ntrain]), 
                       batch_size, epochs)
        
        # since sections are already created at random I might not need to
        #   select the training samples randomly
        
        ### Validation ####
        
        nval = int(len(data)*0.2)
        a, acc = self.model.evaluate(np.array(data[0:nval]), 
                                      np.array(labels[0:nval]),
                                      batch_size=128)
        print("Accuracy: {:4.2f} %".format(acc*100)) 
        
    def predict(self, pic):
        """ Takes a picture and returns an array the boxes considered,
        the predictions for each box and trhe probabilities. """
 
        secs = []
        boxes = utils.make_sections(pic.shape, self.input_shape[0], self.stride)
        for b in boxes:
            x1, y1 = b[0], b[1]
            x2, y2 = b[2], b[3]
            secs.append(pic[y1:y2, x1:x2].flatten() )
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
    