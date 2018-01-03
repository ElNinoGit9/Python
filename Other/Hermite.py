# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:04:43 2016

@author: Markus
"""

import numpy as np


def Hermite(k):
    
    coef = np.zeros(k+1)
    if k == 0:
       coef[0] = 1
    elif k == 1:
        coef[0] = 1
        coef[1] = 1        
    else:
        Cn1 = Hermite(k-1)        
        print(Cn1[1])
        print(Cn2[0])
        
        for n in range(1,k+1):
            coef[n] = Cn1[n] + k*coef[n-1]
        coef[0] = -k*coef[0]
        
    return coef
  
N = 2

Num = Hermite(N)

print(Num)

