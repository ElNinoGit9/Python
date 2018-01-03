# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 22:47:25 2015

@author: Markus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
z = np.power(x,3)
A = np.vstack([x, np.ones(len(x)), z]).T
m = np.linalg.lstsq(A, y)[0]

print(m)