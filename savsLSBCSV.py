# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 14:36:52 2025

@author: Allen
"""

import cv2
import os
import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

def create_labels_csv(root):
    clean_path = root+'//clean'
    stego_path = root+'//stego'
    clean_dir = os.listdir(clean_path)
    stego_dir = os.listdir(stego_path)
    image_name_list = []
    
    for file in clean_dir:
        temp = [0,0]
        temp[0] = file
        image_name_list.append(temp)
        
    for file in stego_dir:
        temp = [0,1]
        temp[0] = file
        image_name_list.append(temp)
        
    return image_name_list

def main():

    test_root = 'C://imcs3010//LSB Dataset//test//test'
    train_root = 'C://imcs3010//LSB Dataset//train//train'
    
    savetxt('lsb_test_labels.csv', asarray(create_labels_csv(test_root)), delimiter=',', fmt="%s")
    savetxt('lsb_train_labels.csv', asarray(create_labels_csv(train_root)), delimiter=',', fmt="%s")
    
    return
    
main()