#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class ResistorGenerator(Sequence):
    
    def __init__(self, batch_size, batches_per_epoch, return_angles,
                 resistor_prob,
                 real_backgrounds,
                 angle_num,
                 flatten,
                 dim = 48
                 ):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        
        self.flatten = flatten
        self.dim = dim
        
        colors = ["beige",
                  "blue",
                  "black",
                  "gold"]
        self.color_imgs = []
        for c in colors:
            fn =  'base_colors/{}_resistor.png'.format(c)
            self.color_imgs.append(Image.open(fn))

            
        
    def __len__(self):
        return self.batches_per_epoch
    
    def __getitem__(self, i):
        """ Returns a batch of randomly generated pictures """
        
        pics = []
        labels = []
        
        for i in range(self.batch_size):
            
            pic = Image.new('RGB', (self.dim, self.dim), (255, 255, 255))
                
            # Base resis
            base_color = 0 # TODO: add more base colors
            base_img = self.color_imgs[base_color]
            pic.paste(base_img, (0, 0, self.dim,self.dim), base_img)
                
            # First ring
                        
            
            first = random.randint(4, 6) # position of the first band
            width = random.randint(2, 4) # width of the bands
            sep = width + random.randint(2, 4) # separation between the bands
            for i in range(0, 4):
                if i < 3:
                    band_color = random.randint(1, len(self.color_imgs)-1)
                    band_pos = first + i*width + i*sep 
                else: # if it's the last ring
                    band_color = -1 # gold
                    band_pos = 40 + random.randint(-2, 2)
                box = (band_pos, 0, band_pos + width, 48)
                band = self.color_imgs[band_color].crop(box)
                pic.paste(band, (band_pos, 0), band)
            
            
            pics.append(np.asarray(pic).reshape((48, 48, 3)))

            labels.append(0)

        return np.array(pics), np.array(labels)
        
if __name__ == "__main__":
    # if called by itself generate five examples
    generator = ResistorGenerator(batch_size = 5, 
                                 batches_per_epoch = 3,
                                 return_angles = True,
                                 resistor_prob = 0.5,
                                 real_backgrounds = True,
                                 angle_num = 8,
                                 flatten=False)
    for r,l in zip(*generator.__getitem__(0)):
#        print(l)
        plt.figure(figsize=(1,1))
        plt.imshow(r, cmap = 'gray' )
        plt.axis("off")
        plt.show()