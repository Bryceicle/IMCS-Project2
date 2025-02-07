# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:55:55 2025

@author: Allen
"""

import os
import cv2 
import numpy as np

def rgbToLSB(image):

    dim = image.shape
    bin_image = np.empty((dim[0],dim[1],dim[2]))
    
    for i in range(0,dim[0]):
        for j in range(0,dim[1]):
            for k in range(0,dim[2]):
                binary = bin(image[i][j][k])[2:].zfill(4) #allows customisation of lsb from 1 to 4
                least_sig_bit = binary[-1:]
                bin_image[i][j][k] = least_sig_bit
    
    return bin_image
            