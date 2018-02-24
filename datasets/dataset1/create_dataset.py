#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:31:46 2018

@author: carles
"""

# Creates a picture with a number of resistors in random positions

import random
from PIL import Image
import numpy as np

from matplotlib import pyplot as plt

base_img = Image.open('resistor.jpg')

dim = 48 # dimensions of the base square image
num = 50 # number of images to be generated
p = 0.3 # fraction of resistors pasted

imgs = [] # images
boxs = [] # positions

for n in range(0, num):
    img = Image.new('F', (240, 240), 'white')
    bs = [] #np.array([]) # boxes for this image
    for i  in range(0, 5):
        for j in range(0, 5):
            if  random.random() < p:
                x = i*48
                y = j*48
                box = (x, y, x + 48, y + 48)
                img.paste(base_img, box)
                bs.extend(box)
    imgs.append(np.asarray(img, dtype=int).flatten())
    boxs.append(bs)
    
    if n < 5:
        plt.figure()
        plt.imshow(np.asarray(img), cmap = 'gray' )

# Fill all rows with -1 to make a rectangular array
maxl = max([len(bs) for bs in boxs])
for bs in boxs:
    bs.extend([-1]*(maxl-len(bs)))
    

np.savetxt("dataset1_imgs.csv", np.array(imgs), delimiter=',', fmt='%d')
np.savetxt("dataset1_boxs.csv", np.array(boxs), delimiter=',', fmt='%d')

        

