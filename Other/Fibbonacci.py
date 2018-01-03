# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:03:00 2015

@author: Markus
"""

import numpy as np


def Fib(N):
    Numbers = np.zeros(N)
    Numbers[0] = 1
    Numbers[1] = 1
    for n in range(2, N):
                Numbers[n] = Numbers[n-1] + Numbers[n-2]
    return Numbers
N = 10

Num = Fib(N)

print(Num)
