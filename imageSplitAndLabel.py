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
import shutil


def datasetPrep(root, binaryClass, target):

    originals_root = root + 'originals/'
    stegos_root = root + 'stegos/'

    if not os.path.exists(target + 'train/originals/'):
        os.mkdir(target + 'train/')
        os.mkdir(target + 'train/originals/')

    if not os.path.exists(target + 'train/stegos/'):
        os.mkdir(target + 'train/stegos/')

    if not os.path.exists(target + 'test/originals/'):
        os.mkdir(target + 'test/')
        os.mkdir(target + 'test/originals/')

    if not os.path.exists(target + 'test/stegos/'):
        os.mkdir(target + 'test/stegos/')

    originals_train = target + 'train/originals/'
    stegos_train = target + 'train/stegos/'
    originals_test = target + 'test/originals/'
    stegos_test = target + 'test/stegos/'

    if os.path.exists(originals_root):
        originals_list = os.listdir(originals_root)
        i = 0
        for file in originals_list:
            if i != 0 and i%3 == 0:
                if not os.path.exists(originals_test+file) and not file.split('.')[1] == 'DNG':
                    shutil.copy2(originals_root+file, originals_test+file)
                    i += 1
            else:
                if not os.path.exists(originals_train+file) and not file.split('.')[1] == 'DNG':
                    shutil.copy2(originals_root+file, originals_train+file)
                    i += 1

    if os.path.exists(stegos_root):
        stegos_list = os.listdir(stegos_root)
        i = 0
        for file in stegos_list:
            if i != 0 and i % 3 == 0:
                if not os.path.exists(stegos_test + file) and not file.split('.')[1] == 'DNG':
                    shutil.copy2(stegos_root + file, stegos_test + file)
                    i += 1
            else:
                if not os.path.exists(stegos_train + file) and not file.split('.')[1] == 'DNG':
                    shutil.copy2(stegos_root + file, stegos_train + file)
                    i += 1

        return
    

def image_split_and_label_binary(path):
    originals = path + 'originals/'
    stegos = path + 'stegos/'

    image_name_list = []
    split_dim = [124, 124]

    if os.path.exists(originals):
        direct = os.listdir(originals)

        for file in direct:
            image = cv2.imread(originals + file)
            tiles = [image[x:x + split_dim[0], y:y + split_dim[1]] for x in
                     range(0, image.shape[0] - (image.shape[0] % split_dim[0]), split_dim[0]) for y in
                     range(0, image.shape[1] - (image.shape[1] % split_dim[1]), split_dim[1])]

            i = 0

            if os.path.exists(path + 'split/'):
                os.chdir(path + 'split/')
            else:
                os.mkdir(path + 'split/')
                os.chdir(path + 'split/')

            for split in tiles:
                temp = [0, 0]
                filename = file[:-4] + 'split' + str(i) + '.png'
                temp[0] = filename
                image_name_list.append(temp)

                if not os.path.isfile(path + 'split/' + filename):
                    cv2.imwrite(filename, split)
                i += 1

    if os.path.exists(stegos):
        direct = os.listdir(stegos)

        for file in direct:
            image = cv2.imread(stegos + file)
            tiles = [image[x:x + split_dim[0], y:y + split_dim[1]] for x in
                     range(0, image.shape[0] - (image.shape[0] % split_dim[0]), split_dim[0]) for y in
                     range(0, image.shape[1] - (image.shape[1] % split_dim[1]), split_dim[1])]

            i = 0

            if os.path.exists(path + 'split/'):
                os.chdir(path + 'split/')
            else:
                os.mkdir(path + 'split/')
                os.chdir(path + 'split/')

            for split in tiles:
                temp = [0, 1]
                filename = file[:-4] + 'split' + str(i) + '.png'
                temp[0] = filename

                if not os.path.isfile(path + 'split/' + filename):
                    cv2.imwrite(filename, split)
                    image_name_list.append(temp)
                i += 1

    return image_name_list

def main():
    target = '/home/bryce/PycharmProjects/IMCS_Datasets/PNG_Only/'

    if not os.path.exists(target):
        os.mkdir(target)

    #root = '/home/bryce/PycharmProjects/IMCS_Datasets/MobiStego/'
    #datasetPrep(root, True, target)
    #root = '/home/bryce/PycharmProjects/IMCS_Datasets/PixelKnot/'
    #datasetPrep(root, True, target)
    #root = '/home/bryce/PycharmProjects/IMCS_Datasets/Passlok/'
    #datasetPrep(root, True, target)
    #root = '/home/bryce/PycharmProjects/IMCS_Datasets/PocketStego/'
    #datasetPrep(root, True, target)
    #root = '/home/bryce/PycharmProjects/IMCS_Datasets/Meznik/'
    #datasetPrep(root, True, target)
    #root = '/home/bryce/PycharmProjects/IMCS_Datasets/Pictograph/'
    #datasetPrep(root, True, target)

    trainRoot = target + 'train/'
    savetxt('lsb_train_labels_binary.csv', asarray(image_split_and_label_binary(trainRoot)), delimiter=',', fmt="%s")

    testRoot = target + 'test/'
    savetxt('lsb_test_labels_binary.csv', asarray(image_split_and_label_binary(testRoot)), delimiter=',', fmt="%s")

    return

main()