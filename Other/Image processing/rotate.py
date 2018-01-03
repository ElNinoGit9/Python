# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 17:43:55 2016

@author: Markus
"""

import numpy as np

def rotate_list(elements, rotates):
    
    end = np.size(elements)
    
    e1 = elements[0:rotates]
    e2 = elements[rotates:end]
    
    elements[0:end-rotates] = e2;
    elements[end-rotates:end] = e1;

    return elements

if __name__ == '__main__':
    assert rotate_list([1, 2, 3, 4, 5, 6], 2) == [3, 4, 5, 6, 1, 2], "First"
    assert rotate_list([1, 2, 3, 4, 5, 6], 3) == [4, 5, 6, 1, 2, 3], "Second"
    assert rotate_list([1, 2, 3, 4, 5, 6], 0) == [1, 2, 3, 4, 5, 6], "Third"
