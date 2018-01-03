# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 22:32:45 2015

@author: Markus
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,np.pi, 0.1)
deg = 3
y = np.sin(x)



c = np.polyfit(x,y,deg)
xx = np.arange(0,np.pi, 0.01)
p = np.polyval(c,xx)

plt.plot(xx,p)
plt.plot(x,y)