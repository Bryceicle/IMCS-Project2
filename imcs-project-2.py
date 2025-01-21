# -*- coding: utf-8 -*-
"""
IMCS Project 2 - Winter 2025
Bryce Gill - 100666638


"""

from tqdm import tqdm
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
from torcheval.metrics.functional import binary_f1_score

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
    else: 
        return 0
    if os.path.exists('C:\imcs3010\IMCS-Project2\stego_lsb_train_data.npy'):
        stego = loadtxt('stego_lsb_train_data.npy', delimiter=',')
    else: 
        return 0
    
    return clean, stego

def load_lsb_arrays(path, flag):
    
    if load_lsb_train_array_from_file() == 0:
        if flag == 'train':
            train_clean_path = path+'//train//train//clean'
            train_stego_path = path+'//train//train//stego'
            
            train_clean_dataset = load_image_dataset_to_array(train_clean_path)
            train_stego_dataset = load_image_dataset_to_array(train_stego_path)
            
            clean_target = [0]*len(train_clean_dataset)
            stego_target = [1]*len(train_stego_dataset)
            
            dataset = asarray(train_clean_dataset + train_stego_dataset)
            targets = clean_target + stego_target
            
            #clean_reshape = clean_dataset.reshape((4000, 786432))
            #stego_reshape = stego_dataset.reshape((12000, 786432))
            train_set = [dataset, targets]
            tensor = torch.tensor(train_set)
            #stego_tensor = torch.tensor(stego_train_dataset)
            #savetxt('clean_lsb_train_data.npy', clean_reshape, delimiter=',')
            #savetxt('stego_lsb_train_data.npy', stego_reshape, delimiter=',')
            #print('saved!')
            
        else:
            test_clean_path = path+'//test//test//clean'
            test_stego_path = path+'//test//test//stego'
            
            test_clean_dataset = load_image_dataset_to_array(test_clean_path)
            test_stego_dataset = load_image_dataset_to_array(test_stego_path)
            
            clean_target = [0]*test_clean_dataset.shape[0]
            stego_target = [1]*test_stego_dataset.shape[0]
            
            dataset = asarray(test_clean_dataset + test_stego_dataset)
            targets = clean_target + stego_target
            
            #clean_reshape = clean_dataset.reshape((4000, 786432))
            #stego_reshape = stego_dataset.reshape((12000, 786432))
            
            train_set = [dataset, targets]
            tensor = torch.tensor(train_set)
            #stego_tensor = torch.tensor(stego_test_dataset)
            #savetxt('clean_lsb_train_data.npy', clean_reshape, delimiter=',')
            #savetxt('stego_lsb_train_data.npy', stego_reshape, delimiter=',')
            #print('saved!')
            
    else:
        train_clean_dataset, train_stego_dataset = load_lsb_train_array_from_file()
        clean_tensor = torch.tensor(train_clean_dataset)
        #stego_tensor = torch.tensor(train_clean_dataset)
        print('loaded!')
        
    return tensor 
    

class LSB_ConvNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(LSB_ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
        
lsb_dataset_path = 'C://imcs3010//LSB Dataset'

BATCH_SIZE = 128
lr = 0.001
NUM_EPOCHS = 3

LSB_Model = LSB_ConvNet(3, 2)

optimizer = optim.SGD(LSB_Model.parameters(), lr=lr, momentum=0.9)

criterion = nn.BCEWithLogitsLoss()
        
def main():
    
    print("Loading Data ...")
    
    LSB_train_dataset = load_lsb_arrays(lsb_dataset_path, 'train')
    train_loader = data.DataLoader(dataset=LSB_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=LSB_train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    print("Loaded LSB training data.")
    
    LSB_test_dataset = load_lsb_arrays(lsb_dataset_path, 'test')
    test_loader = data.DataLoader(dataset=LSB_test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    print("Loaded LSB testing data.")
    
    print("Loading model ...")
    
    for epoch in range(NUM_EPOCHS):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        
        LSB_Model.train()
        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = LSB_Model(inputs)
            
            #if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
            """
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
            """
            
            loss.backward()
            optimizer.step()
            
    # evaluation
    
    def test(split):
        LSB_Model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        
        data_loader = train_loader_at_eval if split == 'train' else test_loader
    
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = LSB_Model(inputs)
    
                #if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
                """
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
                """
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
    
            y_true = y_true.numpy()
            y_score = y_score.detach().numpy()
            
            f1_score = binary_f1_score(outputs, targets)
            """
            if split == 'test':
                eval_metrics_list.append(metrics)
            """
        
            print(f1_score)
            
       
    print('==> Evaluating ...')
    test('train')
    test('test')

    return

main()
    