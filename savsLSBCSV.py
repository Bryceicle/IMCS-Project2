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

def ceildiv(a ,b):
    return -(a //-b)

def image_split_and_label(root):
    clean_path = root+'//clean'
    stego_path = root+'//stego'
    clean_dir = os.listdir(clean_path)
    stego_dir = os.listdir(stego_path)
    
    image_name_list = []
    split_dim = [124,124]
    
    for file in clean_dir:
        image = cv2.imread(clean_path+'//'+file)
        
        tiles = [image[x:x+split_dim[0], y:y+split_dim[1]] for x in range(0, image.shape[0]-(image.shape[0]%split_dim[0]),split_dim[0]) for y in range(0, image.shape[1]-(image.shape[1]%split_dim[1]),split_dim[1])]
        
        i = 0
        
        os.chdir(root+'//split')
        
        for split in tiles:
            filename = file[:-4]+'split'+str(i)+'.png'
            temp=[0,0]
            temp[0] = filename
            image_name_list.append(temp)
                
            if not os.path.isfile(root+'//split//'+filename):
                cv2.imwrite(filename, split)
            i+=1
            
    for file in stego_dir:
        image = cv2.imread(stego_path+'//'+file)
        
        tiles = [image[x:x+split_dim[0], y:y+split_dim[1]] for x in range(0, image.shape[0]-(image.shape[0]%split_dim[0]),split_dim[0]) for y in range(0, image.shape[1]-(image.shape[1]%split_dim[1]),split_dim[1])]
        
        i = 0
        
        os.chdir(root+'//split')
        
        for split in tiles:
            filename = file[:-4]+'split'+str(i)+'.png'
            temp=[0,1]
            temp[0] = filename
            image_name_list.append(temp)
                
            if not os.path.isfile(root+'//split//'+filename):
                cv2.imwrite(filename, split)
            i+=1
    
    os.chdir(root)     
    return image_name_list

def main():

    test_root = 'C://imcs3010//LSB Dataset//test//test'
    train_root = 'C://imcs3010//LSB Dataset//train//train'
    
    savetxt('lsb_test_labels.csv', asarray(image_split_and_label(test_root)), delimiter=',', fmt="%s")
    savetxt('lsb_train_labels.csv', asarray(image_split_and_label(train_root)), delimiter=',', fmt="%s")
    
    return
    
main()