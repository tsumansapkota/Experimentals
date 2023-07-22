###### MODIFIED from 
## https://github.com/chenyaofo/CIFAR-pretrained-models

import torch
import torch.nn as nn
from typing import Union, Tuple
import numpy as np

__all__ = ['CifarResNet', 'cifar_resnet20', 'cifar_resnet32', 'cifar_resnet44', 'cifar_resnet56']

    

class ChannelNorm2D(nn.Module):
    
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        x = x-x.mean(dim=1, keepdim=True)
        x = x/torch.sqrt(x.var(dim=1, keepdim=True)+self.eps)
        return x
    

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1, groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, groups=groups, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1):
        super(BasicBlock, self).__init__()
        
        self.groups = groups
        
        assert planes%groups == 0 , "Layer Planes must be divisible by groups"
        
        
        #################### PROBLEM: WITH DOWNSMAMPLE, THE FIRST BLOCK CAN'T MAKE GROUPS==PLANES 
        #################### SOLUTION 1 #########################
        ## Downsample (x2) with channel double, so incoming plane has less , 
        ## DIVIDE GROUPS IN FIRST BLOCK BY 2 (BECAISE INPLANES ARE MULTIPLE OF 2
#         if downsample and groups > 1:
#             self.groups = int(np.ceil(groups//2))
        
        #################### SOLUTION 2 #########################
        ## Make groups at maximum number of inplanes
        if downsample and groups > inplanes:
            self.groups = inplanes
        
        
        ####### This means do not use grouping technique.. use normal convolution
        if groups == -1:
            self.groups = 1
        #############################################
            
#         print(inplanes, planes, groups, self.groups)
        
        self.conv1 = conv3x3(inplanes, planes, stride, self.groups)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.groups2 = planes//self.groups
        
        ####### This means do not use grouping technique.. use normal convolution
        if groups == -1:
            self.groups2 = 1
            
        #############################################
        
#         print(self.groups2)
        self.conv2 = conv3x3(planes, planes, groups=self.groups2)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)        
        out = self.bn1(out)
        out = self.relu(out)

        B, C, H, W = out.shape
        out = out.view(B, C//self.groups2, self.groups2, H, W)\
                    .transpose(1,2).contiguous()\
                    .view(B, C, H, W)
                      
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, mixer=False, planes=16, G=None):
        super(CifarResNet, self).__init__()
        global conv3x3, conv1x1
        
        ### buffer for last used planes in conv-res-blocks
        self.inplanes = planes
        self.conv1 = conv3x3(3, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        if G is None:
            G = [-1, -1, -1]
            if mixer:
                G = [4, 8, 8]
                
        self.layer1 = self._make_layer(block, planes, layers[0], groups=G[0])
        self.layer2 = self._make_layer(block, planes*2, layers[1], stride=2, groups=G[1])
        self.layer3 = self._make_layer(block, planes*4, layers[2], stride=2, groups=G[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(planes*4 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, groups=1),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ### self.inplanes is from last block of _make_layer_
        layers.append(block(self.inplanes, planes, stride, downsample, groups=groups))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cifar_resnet20(**kwargs):
    model = CifarResNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def cifar_resnet32(**kwargs):
    model = CifarResNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def cifar_resnet44(**kwargs):
    model = CifarResNet(BasicBlock, [7, 7, 7], **kwargs)
    return model


def cifar_resnet56(**kwargs):
    model = CifarResNet(BasicBlock, [9, 9, 9], **kwargs)
    return model