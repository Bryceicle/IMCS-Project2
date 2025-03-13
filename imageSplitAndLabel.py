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
from sympy.strategies.core import switch


def datasetPrep(root, binaryClass, dataset):

    if dataset == 0:
        train_root = root + 'train/train/'
        test_root = 'test/test/'

        if binaryClass:
            if os.path.exists(test_root) and not os.path.isfile(test_root + 'lsb_test_labels_binary.csv'):
                clean_path = test_root + '/clean/'
                stego_path = test_root + '/stego/'
                os.chdir(clean_path)
                savetxt('lsb_test_labels_binary.csv', asarray(image_split_and_label_binary(clean_path)), delimiter=',',
                        fmt="%s")
                os.chdir(stego_path)
                savetxt('lsb_test_labels_binary.csv', asarray(image_split_and_label_binary(stego_path)), delimiter=',',
                        fmt="%s")
            if os.path.exists(train_root) and not os.path.isfile(train_root + 'lsb_train_labels_binary.csv'):
                clean_path = train_root + '/clean/'
                stego_path = train_root + '/stego/'
                os.chdir(clean_path)
                savetxt('lsb_train_labels_binary.csv', asarray(image_split_and_label_binary(clean_path)), delimiter=',',
                        fmt="%s")
                os.chdir(stego_path)
                savetxt('lsb_train_labels_binary.csv', asarray(image_split_and_label_binary(stego_path)), delimiter=',',
                        fmt="%s")
        else:
            if os.path.exists(test_root) and not os.path.isfile(test_root + 'lsb_test_labels_multiclass.csv'):
                clean_path = test_root + '/clean/'
                stego_path = test_root + '/stego/'
                os.chdir(clean_path)
                savetxt('lsb_test_labels_multiclass.csv', asarray(image_split_and_label_multiclass(clean_path)),
                        delimiter=',', fmt="%s")
                os.chdir(stego_path)
                savetxt('lsb_test_labels_multiclass.csv', asarray(image_split_and_label_multiclass(stego_path)),
                        delimiter=',', fmt="%s")
            if os.path.exists(train_root) and not os.path.isfile(train_root + 'lsb_train_labels_multiclass.csv'):
                clean_path = train_root + '/clean/'
                stego_path = train_root + '/stego/'
                os.chdir(clean_path)
                savetxt('lsb_test_labels_multiclass.csv', asarray(image_split_and_label_multiclass(clean_path)),
                        delimiter=',', fmt="%s")
                os.chdir(stego_path)
                savetxt('lsb_test_labels_multiclass.csv', asarray(image_split_and_label_multiclass(stego_path)),
                        delimiter=',', fmt="%s")

    else:
        originals_root = root + 'originals/'
        stegos_root = root + 'stegos/'
        trainRoot = root + 'train/'
        testRoot = root + 'test/'
        if not os.path.exists(trainRoot):
            os.mkdir(trainRoot)
        if not os.path.exists(testRoot):
            os.mkdir(testRoot)

        for file in originals_root:

            os.system('cp source.txt destination.txt')

        if os.path.exists(originals_root) and not os.path.isfile(originals_root+'lsb_original_labels_binary.csv'):
            savetxt('lsb_originals_labels_binary.csv', asarray(image_split_and_label_binary(originals_root)), delimiter=',',fmt="%s")
        if os.path.exists(stegos_root) and not os.path.isfile(stegos_root+'lsb_original_labels_binary.csv'):
            savetxt('lsb_stegos_labels_binary.csv', asarray(image_split_and_label_binary(stegos_root)), delimiter=',',fmt="%s")

    return

def image_split_and_label_binary(path):
    image_name_list = []
    split_dim = [124, 124]

    if os.path.exists(path):
        direct = os.listdir(path)

        for file in direct:
            image = cv2.imread(path + file)

            tiles = [image[x:x + split_dim[0], y:y + split_dim[1]] for x in
                     range(0, image.shape[0] - (image.shape[0] % split_dim[0]), split_dim[0]) for y in
                     range(0, image.shape[1] - (image.shape[1] % split_dim[1]), split_dim[1])]

            i = 0

            if os.path.exists(path + '/split'):
                os.chdir(path + '/split')
            else:
                os.mkdir(path + '/split')
                os.chdir(path + '/split')

            for split in tiles:
                filename = file[:-4] + 'split' + str(i) + '.png'
                temp = [0, 0]
                temp[0] = filename
                image_name_list.append(temp)

                if not os.path.isfile(path + 'split/' + filename):
                    cv2.imwrite(filename, split)
                i += 1

    return image_name_list

def image_split_and_label_multiclass(path):
    image_name_list = []
    split_dim = [124,124]

    if os.path.exists(path):
        direct = os.listdir(path)
    
    for file in direct:
        image = cv2.imread(direct+file)
        
        tiles = [image[x:x+split_dim[0], y:y+split_dim[1]] for x in range(0, image.shape[0]-(image.shape[0]%split_dim[0]),split_dim[0]) for y in range(0, image.shape[1]-(image.shape[1]%split_dim[1]),split_dim[1])]
        
        i = 0
        
        os.chdir(direct+split)
        
        for split in tiles:
            filename = file[:-4]+'split'+str(i)+'.png'
            temp=[0,0]
            temp[0] = filename
            image_name_list.append(temp)
                
            if not os.path.isfile(direct+'split/'+filename):
                cv2.imwrite(filename, split)
            i+=1

    return image_name_list

def main():
    dataset = 0
    linux = True
    if linux:
        match dataset:
            case 0:
                root = '/home/bryce/PycharmProjects/IMCS Datasets/marcozuppelli/'
                datasetPrep(root, True)

            case 1:
                root = '/home/bryce/PycharmProjects/IMCS Datasets/MobiStego_S8_0-10_auto/'


    return

main()