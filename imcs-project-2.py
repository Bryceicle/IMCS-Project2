# -*- coding: utf-8 -*-
"""
IMCS Project 2 - Winter 2025
Bryce Gill - 100666638


"""

import cv2
import numpy as np
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

def load_image_dataset_to_array(path):
    dir_list = os.listdir(path)
    image_list = []
    
    for file in dir_list:
        image = cv2.imread(path+'//'+file)
        image_list.append(image)
        
    return image_list

def load_lsb_train_array_from_file():
    
    if os.path.exists('C:\imcs3010\IMCS-Project2\clean_lsb_train_data.npy'):
        clean = loadtxt('clean_lsb_train_data.npy', delimiter=',')
        print('found')
    else: 
        print('not found')
        return 0
    if os.path.exists('C:\imcs3010\IMCS-Project2\stego_lsb_train_data.npy'):
        stego = loadtxt('stego_lsb_train_data.npy', delimiter=',')
    else: 
        return 0
    
    return clean, stego

def load_lsb_train_arrays(path):
    
    if load_lsb_train_array_from_file() == 0:
        train_clean_path = path+'//train//train//clean'
        train_stego_path = path+'//train//train//stego'
        train_clean_dataset = load_image_dataset_to_array(train_clean_path)
        train_stego_dataset = load_image_dataset_to_array(train_stego_path)
        clean_dataset = asarray(train_clean_dataset)
        stego_dataset = asarray(train_stego_dataset)
        savetxt('clean_lsb_train_data.npy', clean_dataset, delimiter=',')
        savetxt('stego_lsb_train_data.npy', stego_dataset, delimiter=',')
        print('saved!')
    else:
        train_clean_dataset, train_stego_dataset = load_lsb_train_array_from_file()
        print('loaded!')
        
    return train_clean_dataset, train_stego_dataset
    

class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()
        
lsb_dataset_path = 'C://imcs3010//LSB Dataset'
        
def main():
    
    train_clean_dataset, train_stego_dataset = load_lsb_train_arrays(lsb_dataset_path)
    
    return

main()
    