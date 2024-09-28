import requests
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import math
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

def sap_unit(data, frac):
    act = data.view(data.size(0), -1) # 第一维是batchsize
    
    prob = torch.abs(act) / (torch.abs(act)).sum(dim=1, keepdim=True)
    
    batch_size = data.size(0)
    num_features = act.size(1) # 标量的个数
    num_pruned = int(num_features * frac) # 需要剪枝的个数
    
    # Get indices of the top (1 - frac) features
    _, indices = torch.topk(prob, num_features - num_pruned, dim=1)
    
    # Create a mask
    mask = torch.zeros_like(act, device=act.device)
    mask.scatter_(dim=1, index=indices, value=1)
    
    # Apply the mask to the original activations
    pruned_act = act * mask # element-wise multiply
    
    # Reshape back to original data shape
    return pruned_act.view(data.size())



class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dim_match=True, typ='normal', frac=1.0):
        super(ResidualUnit, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        
        if not dim_match:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        
        self.typ = typ
        self.frac = frac

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        
        if self.typ == 'sap':
            out = sap_unit(out, self.frac)
        
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)

        if self.typ == 'sap':
            out = sap_unit(out, self.frac)
            
        out = self.conv2(out)
        return out + self.shortcut(x)  # 返回残差连接结果

class ResNet(nn.Module):
    def __init__(self, typ, frac):
        super(ResNet, self).__init__()
        self.typ = typ
        self.frac = frac
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            ResidualUnit(16, 16, stride=1, typ=self.typ, frac=self.frac, dim_match=False),
            ResidualUnit(16, 16, stride=1, typ=self.typ, frac=self.frac, dim_match=True),
            ResidualUnit(16, 16, stride=1, typ=self.typ, frac=self.frac, dim_match=True)
        )
        self.layer2 = nn.Sequential(
            ResidualUnit(16, 32, stride=2, typ=self.typ, frac=self.frac, dim_match=False),
            ResidualUnit(32, 32, stride=1, typ=self.typ, frac=self.frac, dim_match=True),
            ResidualUnit(32, 32, stride=1, typ=self.typ, frac=self.frac, dim_match=True)
        )
        self.layer3 = nn.Sequential(
            ResidualUnit(32, 64, stride=2, typ=self.typ, frac=self.frac, dim_match=False),
            ResidualUnit(64, 64, stride=1, typ=self.typ, frac=self.frac, dim_match=True),
            ResidualUnit(64, 64, stride=1, typ=self.typ, frac=self.frac, dim_match=True)
        )
        # self.layer1 = self._make_layer(block, 16, num_blocks[0])
        # self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, typ=self.typ))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, typ=self.typ))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.bn1(x)
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn2(x)

        x = self.relu(x)
        if self.typ == 'sap':
            relu1 = sap_unit(relu1, self.frac)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x