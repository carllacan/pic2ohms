#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 18:20:33 2018

@author: carles
"""

# This script uses the localizer, goniometer and reader modules to
# find, rotate and read resistors in an image.

import numpy as np
import random

from localizer import Localizer
from goniometer import Goniometer

# TODO: showcase localizer and goniometer
# load the best localizer and goniometer
# load an image 
# make a b/w version
# use  localizer to find resistors
# pass the subimages to goniometer to read the angles
# get the subimages from the color picture
# rotate the subimages back to 0º and output them.

# TODO: implement the reader and use it to tag the subimages