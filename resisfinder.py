#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:56:17 2018

@author: carles
"""

# As a first step, the simplest possible case:
    
# A neural network that takes a 48x48 picture and decides whether
# there is a resistor or not. Background is uniform.

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2, l1
from keras.utils.np_utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np