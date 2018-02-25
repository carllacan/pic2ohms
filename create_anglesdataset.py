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

# Creates set of pictures of resistors with diferent color rings

# (not yet implemented)

import random
from PIL import Image
import numpy as np

from matplotlib import pyplot as plt

base_img = Image.open('base_pics/resistor1.png')

dataset = 6

dim = 48 # dimensions of the base square image
num = 5 # number of images to be generated

ring1_colors = range(9)
ring2_colors = range(10)
ring3_colors = range(12)
ring4_colors = range(2)  # but actually 8 colors

pics = []
values = []
for n in range(0, num):
    bgcolor = random.randint(250, 255)
    pic = Image.new('RGB', (dim, dim), bgcolor)
    box = (0, 0, dim, dim)
    ring1 = random.choice(ring1_colors)
    ring2 = random.choice(ring2_colors)
    ring3 = random.choice(ring3_colors)
    ring4 = random.choice(ring4_colors)
    
    pic.paste(pic, box, pic)
    # paste images of the appropriate rings
    # we would need to get resistors of the appropriate colors
    
    pics.append(np.asarray(pic, dtype=int).flatten())
    values.append((ring1, ring2, ring3, ring4))
    if n < 5:
        plt.figure()
        plt.imshow(np.asarray(pic), cmap = 'gray' )
        pic.save('datasets/dataset{0}/example{1}.png'.format(
                dataset,n))
        
np.savetxt("datasets/dataset{0}/dataset{0}_imgs.csv".format(dataset), 
           np.array(pics), delimiter=',', fmt='%d')
np.savetxt("datasets/dataset{0}/dataset{0}_values.csv".format(dataset), 
           np.array(values), delimiter=',', fmt='%d')
