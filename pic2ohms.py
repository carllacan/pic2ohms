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