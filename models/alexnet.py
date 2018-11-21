# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 01:04:18 2018

@author: Mohamed
"""

import torch
import torch.nn as nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten , self).__init__()
    def forward(self , input):
        return input.view(input.shape[0] , -1)
    

class AlexNet:
    def __init__(self , num_classes , learning_rate):
        self.net = nn.Sequential(
            
            nn.Conv2d(in_channels = 3 , out_channels = 96 , kernel_size = 11 , stride = 4 , padding = 4),
            nn.LocalResponseNorm(size = 5 , k = 2),
            nn.ReLU(inplace = True),
            
            nn.MaxPool2d(kernel_size = 3 , stride = 2 , padding = 1),
            
            nn.Conv2d(in_channels = 96 , out_channels = 196 , kernel_size = 5 , stride = 1 , padding = 2),
            nn.LocalResponseNorm(size = 5 , k = 2),
            nn.ReLU(inplace = True),
            
            nn.MaxPool2d(kernel_size = 3 , stride = 2 , padding = 1),
            
            nn.Conv2d(in_channels = 196 , out_channels = 256 , kernel_size = 3 , stride = 1 , padding = 1),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(in_channels = 256 , out_channels = 384 , kernel_size = 3 , stride = 1 , padding = 1),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(in_channels = 384 , out_channels = 256 , kernel_size = 3 , stride = 1 , padding = 1),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(in_channels = 256 , out_channels = 256 , kernel_size = 3 , stride = 1 , padding = 1),
            nn.ReLU(inplace = True),
            
            Flatten(),
            
            nn.Linear(in_features = 256 * 14 * 14 , out_features = 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(inplace = False),
            
            nn.Linear(in_features = 4096 , out_features = 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(inplace = True),
            
            nn.Linear(in_features = 4096 , out_features = num_classes)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters() , lr = learning_rate , momentum = 0.9)