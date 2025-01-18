# -*- coding: utf-8 -*-
"""
IMCS Project 2 - Winter 2025
Bryce Gill - 100666638


"""

import cv2
from docopt import docopt
import numpy as np
import os

def load_image_dataset(path):
    dir_list = os.listdir(path)
    image_list = []
    
    for file in dir_list:
        image = cv2.imread(path+'//'+file)
        image_list.append(image)
        
    return image_list

train_clean_path = 'C://imcsProject2//Stego-Images-Dataset-(LSB)//train//train//clean'
train_stego_path = 'C://imcsProject2//Stego-Images-Dataset-(LSB)//train//train//stego'

train_clean_dataset = load_image_dataset(train_clean_path)
train_stego_dataset = load_image_dataset(train_stego_path)

