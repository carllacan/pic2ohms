#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

class ResistorGenerator(Sequence):
    
    def __init__(self, batch_size, batches_per_epoch,
                 labeled_band,
                 dim = 48):
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        
        self.dim = dim
#        
#        colors = ["Beig",
#                  "Roig",
#                  "Taronja",
#                  "Verd",
#                  "Violeta",
#                  "Dorat"]
        colors = ["beige",
                  "red",
#                  "yellow",
#                  "green",
                  "blue",
                  "gold"]
        self.color_imgs = []
        for c in colors:
#            fn =  'base_colors/Colorins/{}.png'.format(c)
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
            sep = width + random.randint(2, 2) # separation between the bands
            bands = []
            for i in range(0, 4):
                if i < 3:
                    band_color = random.randint(1, len(self.color_imgs)-2)
                    band_pos = first + i*width + i*sep 
                    bands.append(band_color)
                else: # if it's the last ring
                    band_color = -1 # gold
                    band_pos = 40 + random.randint(-2, 2)
                box = (band_pos, 0, band_pos + width, 48)
                band = self.color_imgs[band_color].crop(box)
                pic.paste(band, (band_pos, 0), band)
            
            
            pics.append(np.asarray(pic).reshape((48, 48, 3)))
            label = to_categorical(bands[0]-1, len(self.color_imgs) - 2)
            labels.append(label)

        return np.array(pics), np.array(labels)
        
if __name__ == "__main__":
    # if called by itself generate five examples
    generator = ResistorGenerator(batch_size = 5, 
                                 batches_per_epoch = 3,
                                 labeled_band=1)
    for r,l in zip(*generator.__getitem__(0)):
        print(l)
        plt.figure(figsize=(1,1))
        plt.imshow(r, cmap = 'gray' )
        plt.axis("off")
        plt.show()