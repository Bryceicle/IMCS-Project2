# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:55:55 2025

@author: Allen
"""

import os
import cv2 
import numpy as np

def rgbToLSB(path):
    
    clean_image = cv2.imread(path)
    dim = clean_image.shape
    bin_image = np.empty((dim[0],dim[1],dim[2]))
    
    for i in range(0,dim[0]):
        for j in range(0,dim[1]):
            for k in range(0,dim[2]):
                binary = bin(clean_image[i][j][k])[2:].zfill(4)
                least_sig_bit = binary[-1:]
                bin_image[i][j][k] = least_sig_bit
    
    return bin_image
            
rgbToLSB("C://imcsProject2//Stego-Images-Dataset-(LSB)//test//test//clean//04002.png")