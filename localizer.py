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

import numpy as np
import random
import utils
import h5py 

class Localizer():
    
    def __init__(self, **params):

        self.picshape = params['picshape']
        
        hidden_layers = params['hidden_layers']
        rel = lambda: l2(0.2)
        self.sec_dim = 48
        self.stride = self.sec_dim//2#int(self.sec_dim/3)
        self.model = Sequential()
        for l in hidden_layers:
            self.model.add(Dense(l, 
                                 input_dim = self.sec_dim**2, 
                                 kernel_initializer="uniform", 
                                 activation="relu", 
                                 kernel_regularizer=rel()))
        self.model.add(Dense(2, 
                             kernel_initializer="uniform", 
                             activation="softmax",
                             kernel_regularizer=rel()))
                        
        self.model.compile(loss="categorical_crossentropy", 
                           optimizer="adam", metrics=['accuracy'])
        
    def train(self, pics, boxs, epochs, batch_size):
        
        # Make the training data
        picn = pics.shape[0] # number of pics
        data_size = picn*25 # sections to create
        data = []
        labels = []
        
        sec_dim = self.sec_dim # section size
        stride = self.stride # section stride
        
        for n in range(data_size):
            # Get a random image index
            pic_ind = random.randint(0, picn-1)
            pic = pics[pic_ind,:].reshape(self.picshape)
        
            secs = utils.make_sections(pic.reshape(self.picshape), 
                                       sec_dim, stride)
            sec = random.choice(secs) # choosen section
            label = 0
            for b in boxs[pic_ind,:]:
                # If the bsection exactly contains a resistor mark as 1
                # I may need to relax this definition of error later on
                # Perhaps use the overlapping area with actual resistors
                if (sec == b).all():
                    label = 1
                    
            x1, y1 = sec[0], sec[1]
            x2, y2 = sec[2], sec[3]
            cropped = pic.reshape(self.picshape)[y1:y2, x1:x2].flatten()
            data.append(cropped)
            labels.append(to_categorical(label, 2))
            
        #### Training ####
#        print(data[0:5])
        ntrain = int(data_size*0.8)
#        epochs = 15
#        batch_size = 15
        self.model.fit(np.array(data[0:ntrain]), 
                            np.array(labels[0:ntrain]), 
                            batch_size, epochs)
        
        # since sections are already created at random I might not need to
        #   select the training samples randomly
        
        ### Validation ####
        
        nval = int(data_size*0.2)
        a, acc = self.model.evaluate(np.array(data[0:nval]), 
                                      np.array(labels[0:nval]),
                                      batch_size=128)
        print("Accuracy: {:4.2f} %".format(acc*100))

    def localize1(self, pic, probs=False):
        """ Takes a picture and returns an array of resistor boxes"""
        
        found = []
        boxes = utils.make_sections(pic, self.sec_dim, self.stride)
        for b in boxes:
            x1, y1 = b[0], b[1]
            x2, y2 = b[2], b[3]
            sec = pic[y1:y2, x1:x2].flatten() 
            pred = self.model.predict_classes(np.array([sec]))[0]
            if pred == 1:
                found.append(b)
            probs.append(self.model.predict(np.array([sec]))[0])

        return 
    
    def predict(self, pic):
        """ Takes a picture and returns an array of resistor boxes"""
        
        secs = []
        boxes = utils.make_sections(pic, self.sec_dim, self.stride)
        for b in boxes:
            x1, y1 = b[0], b[1]
            x2, y2 = b[2], b[3]
            secs.append(pic[y1:y2, x1:x2].flatten() )
        preds = self.model.predict_classes(np.array(secs))
        probs = self.model.predict(np.array(secs))

        return boxes, preds, probs
    
    def localize(self, pic):
        boxes, preds, probs = self.predict(pic)
        found = []
        for i in range(len(boxes)):
            if preds[i]:
                found.append(boxes[i])
                
        return found
    
    def save(self, f):
        """Saves current parameters of the model to a file """
        self.model.save(f)