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

def image_split_and_label(root, multiclass):
    
    test_root = root + '//test//test'
    train_root = root + '//train//train'
    
    if multiclass == 0:
        savetxt('lsb_test_labels_binary.csv', asarray(image_split_and_label_binary(test_root)), delimiter=',', fmt="%s")
        savetxt('lsb_train_labels_binary.csv', asarray(image_split_and_label_binary(train_root)), delimiter=',', fmt="%s")
    else:
        savetxt('lsb_test_labels_multiclass.csv', asarray(image_split_and_label_multiclass(test_root)), delimiter=',', fmt="%s")
        savetxt('lsb_train_labels_multiclass.csv', asarray(image_split_and_label_multiclass(train_root)), delimiter=',', fmt="%s")
    
    return

def image_split_and_label_binary(root):
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

def image_split_and_label_multiclass(root):
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
            img_class = file.split('_')[2]
            match img_class:
                case "eth":
                    temp=[0,1]
                case "ps":
                    temp=[0,2]
                case "html":
                    temp=[0,3]
                case "url":
                    temp=[0,4]
                case "js":
                    temp=[0,5]
            temp[0] = filename
            image_name_list.append(temp)
                
            if not os.path.isfile(root+'//split//'+filename):
                cv2.imwrite(filename, split)
            i+=1
    
    os.chdir(root)     
    return image_name_list
