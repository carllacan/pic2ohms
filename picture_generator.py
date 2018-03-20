#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class PictureGenerator(Sequence):
    
    def __init__(self, batch_size, batches_per_epoch, return_angles,
                 resistor_prob,
                 real_backgrounds,
                 angle_num,
                 flatten,
                 dim = 48
                 ):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        # this generator can be used to train either the localizer or the
        # goniometer model, so it must be told what to return
        self.return_angles = return_angles 
        self.resistor_prob = resistor_prob
        self.real_backgrounds = real_backgrounds
        self.flatten = flatten
        self.dim = dim
        
        self.initial_angle = 45 # angle the base resistor is oriented at
        self.angle_num = angle_num
        self.angles = range(0, 360, 360//self.angle_num)
        
        resistor_num = 1
        self.resistors = []
        for i in range(resistor_num):
            fn =  'base_pics/resistor{}.png'.format(i)
            img =  Image.open(fn)
            img = img.resize((self.dim, self.dim))
            # add a rotated version for each angle, better than rotating in real-time
#            self.resistors.append([img.rotate(a) for a in self.angles])
            d = {ang: img.rotate(ang) for ang in self.angles}
            self.resistors.append(d)

            
        background_num = 6        
        self.backgrounds = []
        for i in range(background_num):
            fn =  'base_bgs/bg{}.jpg'.format(i)
            img =  Image.open(fn)
            img = img.convert(mode='L')
            self.backgrounds.append(img)
            
        
    def __len__(self):
        return self.batches_per_epoch
    
    def __getitem__(self, i):
        """ Returns a batch of randomly generated pictures """
        
        pics = []
        labels = []
        
        for i in range(self.batch_size):
            
            if self.real_backgrounds:
                # can I use choice here?
                bg = random.choice(self.backgrounds)
                crop_x = random.randint(0, bg.size[0]//self.dim)
                crop_y = random.randint(0, bg.size[1]//self.dim)
                crop_box = (crop_x, crop_y, 
                            crop_x + self.dim, crop_y + self.dim)
                pic = bg.crop(crop_box)
            else:   
                bgcolor = random.randint(120, 255)
                pic = Image.new('L', (self.dim, self.dim), bgcolor)
                
            resistor = random.random() > self.resistor_prob
            if resistor or self.return_angles:
                # no need to generate empty pics if training goniometer
                angle = random.choice(self.angles)
                resis_img = random.choice(self.resistors) [angle]
                
#                resis_img = resis_img.rotate(angle)
                pic.paste(resis_img, (0, 0, self.dim,self.dim), resis_img)
            else:
                # TODO add non-centered resistors?
                pass
                
            if self.flatten:
                pics.append(np.asarray(pic).flatten())
            else:
                pics.append(np.asarray(pic).reshape((48, 48, 1)))

            if not self.return_angles:
                labels.append(to_categorical(resistor, 2))
            else:
                real_angle = (angle + self.initial_angle) % 360
                labels.append(real_angle)

        return np.array(pics), np.array(labels)
        
if __name__ == "__main__":
    # if called by itself generate five examples
    generator = PictureGenerator(batch_size = 5, 
                                 batches_per_epoch = 3,
                                 return_angles = True,
                                 resistor_prob = 0.5,
                                 real_backgrounds = True,
                                 angle_num = 8,
                                 flatten=False)
    for r,l in zip(*generator.__getitem__(0)):
        print(l)
        plt.figure(figsize=(1,1))
        plt.imshow(r.reshape(generator.dim, generator.dim), cmap = 'gray' )
        plt.axis("off")
        plt.show()