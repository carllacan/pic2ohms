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

base_img = Image.open('base_pics/resistor1.png')

dataset = 4

dim = 48 # dimensions of the base square image
num = 200 # number of images to be generated
pic_shape = 240, 240
p = 0.3 # fraction of resistors pasted
angles = list(range(0, 360, 45))
xs = range(0, pic_shape[0], pic_shape[0]//5//2)
ys = range(0, pic_shape[0], pic_shape[0]//5//2)
pics = [] # images
boxs = [] # positions

        
for n in range(0, num):
    bgcolor = random.randint(120, 255)
    pic = Image.new('L', (240, 240), bgcolor)
    bs = []
    c = int(p*pic_shape[0]*pic_shape[1]/dim**2)
    for i  in range(0, c):
        x = random.choice(xs)
        y = random.choice(ys)
        box = (x, y, x + dim, y + dim)
        img = base_img.rotate(random.choice(angles))
        pic.paste(img, box, img)
        bs.extend(box)
    pics.append(np.asarray(pic, dtype=int).flatten())
    boxs.append(bs)
    
    if n < 5:
        plt.figure()
        plt.imshow(np.asarray(pic), cmap = 'gray' )
        pic.save('datasets/dataset{0}/example{1}.png'.format(
                dataset,n))

# Fill all rows with -1 to make a rectangular array
maxl = max([len(bs) for bs in boxs])
for bs in boxs:
    bs.extend([-1]*(maxl-len(bs)))
    
np.savetxt("datasets/dataset{0}/dataset{0}_imgs.csv".format(dataset), 
           np.array(pics), delimiter=',', fmt='%d')
np.savetxt("datasets/dataset{0}/dataset{0}_boxs.csv".format(dataset), np.array(boxs), 
           delimiter=',', fmt='%d')

        

