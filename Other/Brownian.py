# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:33:35 2015

@author: Markus
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.9, 2.1])
z = np.power(x, 3)
A = np.vstack([x, np.ones(len(x)), z]).T
m = np.linalg.lstsq(A, y)[0]
tMin = 0
tMax = 100
dt = 0.0001
t = np.arange(0, 1, dt)
Nt = len(t)
mu, sigma = 0, 1  # mean and standard deviation
s = np.random.normal(mu, sigma, Nt)

W = np.cumsum(s)/np.sqrt(Nt)
t = t*tMax
W = W * np.sqrt(tMax)

'''Y = (mu-(sigma^2)/2)*t + sigma * W;

X = np.exp(Y);'''
plt.plot(t, W)
