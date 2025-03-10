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
import torchvision.transforms as transform
import torchvision.transforms.functional as transform_functional
from torcheval.metrics.functional import binary_f1_score
from torcheval.metrics.functional import multiclass_f1_score
import pandas as pd

class LSB_Dataset(data.Dataset):
    
    def __init__(self, labels_file, img_dir, image_transform=None, target_transform=None):
        
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.image_transform = image_transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index,0])
        image = cv2.imread(img_path) 
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        image = image.float()
        if self.img_labels.iloc[index,1] == 0:
            label = torch.tensor([0])
        else:
            label = torch.tensor([1])
        if self.image_transform:
            image = self.image_transform(image) 
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[2], stride=2)
        #self.layer3 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
train_root_dir = '/home/bryce/PycharmProjects/IMCS Datasets/marcozuppelli/'
test_root_dir = '/home/bryce/PycharmProjects/IMCS Datasets/marcozuppelli/'

BATCH_SIZE = 128
lr = 0.001
NUM_EPOCHS = 3

num_classes = 1

LSB_Model = ResNet(ResidualBlock, [2,2,2], num_classes)

optimizer = optim.SGD(LSB_Model.parameters(), lr=lr, momentum=0.9)

if num_classes == 1:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    
#image_transform=transform.Normalize(0.5, 0.5)
        
def main():
    
    print("==> Loading Data ...")
    
    if num_classes == 1:
        label_class = 'binary'
    else:
        label_class = 'multiclass'
    
    train_labels = train_root_dir+'/train/train/lsb_train_labels_'+label_class+'.csv'
    test_labels = test_root_dir+'/test/test/lsb_test_labels_'+label_class+'.csv'
    
    train_data = LSB_Dataset(train_labels, train_root_dir+'/train/train/split')
    test_data = LSB_Dataset(test_labels, test_root_dir+'/test/test/split')
      
    train_dataloader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    train_dataloader_at_eval = data.DataLoader(train_data, batch_size=2*BATCH_SIZE, shuffle=False)
    test_dataloader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    load_model = True

    if not load_model:

        print("Done.\n==> Training model...")

        for epoch in range(NUM_EPOCHS):

            LSB_Model.train()
            for inputs, targets in tqdm(train_dataloader):
                # forward + backward + optimize
                optimizer.zero_grad()
                outputs = LSB_Model(inputs)

                if num_classes == 1:
                    targets = targets.to(torch.float32)
                    loss = criterion(outputs, targets)
                else:
                    targets = targets.squeeze().long()
                    loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

        torch.save(LSB_Model.state_dict(), '/home/bryce/PycharmProjects/LSB_ResNet.pth')

    else:
        print("Done.\n==> Loading model...")
        LSB_Model.load_state_dict(torch.load('/home/bryce/PycharmProjects/LSB_ResNet.pth'))
            
    # evaluation
    
    def test(split):
        
        false_positives = []
        false_negatives = []
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        
        
        LSB_Model.eval()
        
        data_loader = train_dataloader_at_eval if split == 'train' else test_dataloader
    
        with torch.no_grad():
            for inputs, targets in tqdm(data_loader):
                outputs = LSB_Model(inputs)
    
                if num_classes == 1:
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
                """
                for i in range(len(outputs)):
                    if outputs[i] > 0.5 and targets[i] == 0:
                        false_positives.append(inputs[1])
                    if outputs[i] < 0.5 and targets[i] == 1:
                        false_negatives.append(inputs[1])
                """  
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
    
            #y_true = y_true.numpy()
            #y_score = y_score.detach().numpy()
            
            y_true = y_true.squeeze()
            y_score = y_score.squeeze()
            
            if num_classes == 1:
                f1_score = binary_f1_score(y_score, y_true)
            else:
                f1_score = multiclass_f1_score(y_score, y_true, num_classes=num_classes)
            
            acc = (torch.argmax(y_score) == torch.argmax(y_true)).float().mean()
                
            print('\nacc: ', acc)
            print('f1: ', f1_score)
            
       
    print('==> Evaluating ...')
    print('Training')
    test('train')
    print('\nTesting:')
    test('test')


    return



main()
    