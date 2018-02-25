#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 10:50:25 2018

@author: carles
"""

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

dataset = 6

dim = 48 # dimensions of the base square image
num = 2000 # number of images to be generated

angle_list = list(range(0, 360, 45))
pics = []
angles = []

for n in range(0, num):
    bgcolor = random.randint(50, 255)
    pic = Image.new('L', (dim, dim), bgcolor)
    box = (0, 0, dim, dim)
    angle = random.choice(angle_list)
    img = base_img.rotate(angle)
    pic.paste(img, box, img)
    pics.append(np.asarray(pic, dtype=int).flatten())
    angles.append((angle+45)%360)
    if n < 5:
        plt.figure()
        plt.imshow(np.asarray(pic), cmap = 'gray' )
        pic.save('datasets/dataset{0}/example{1}.png'.format(
                dataset,n))
        
np.savetxt("datasets/dataset{0}/dataset{0}_imgs.csv".format(dataset), 
           np.array(pics), delimiter=',', fmt='%d')
np.savetxt("datasets/dataset{0}/dataset{0}_angles.csv".format(dataset), 
           np.array(angles), delimiter=',', fmt='%d')
