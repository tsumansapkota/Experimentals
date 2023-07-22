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
        self.conv1 = conv3x3(inplanes, planes, stride, self.groups)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
#         self.groups2 = planes//self.groups
        self.conv2 = conv3x3(planes, planes, groups=self.groups)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)        
        out = self.bn1(out)
        out = self.relu(out)

#         B, C, H, W = out.shape
#         out = out.view(B, C//self.groups2, self.groups2, H, W)\
#                     .transpose(1,2).contiguous()\
#                     .view(B, C, H, W)
                      
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SequentialMixer(nn.Module):
    
    def __init__(self, blocks, inplanes, group_size):
        super().__init__()
        
        self.blocks = nn.ModuleList(blocks)
        self.inplanes = inplanes
        self.group_sz = group_size
        self.groups = inplanes//group_size
        
        def log_base(a, base):
            return np.log(a) / np.log(base)
        
        
        ### total number of layers to complete mixing
        self.num_layers = int(np.ceil(log_base(self.inplanes, base=self.group_sz)))
        
        self.gaps = []
        for i in range(len(self.blocks)):
            butterfly_layer_index = i%self.num_layers ## repeated index in blocks (for layers)

            gap = self.group_sz**butterfly_layer_index
            if gap*self.group_sz > self.inplanes:
                gap = int(np.ceil(self.inplanes/self.group_sz))
            self.gaps += [gap]
            pass
        
        
        pass
    
    def forward(self, x):
        
        B, C, H, W = x.shape
#         out = out.view(B, C//self.groups2, self.groups2, H, W)\
#                     .transpose(1,2).contiguous()\
#                     .view(B, C, H, W)

        for gap, fn in zip(self.gaps, self.blocks):
#         for i, fn in enumerate(self.blocks):
#             butterfly_layer_index = i%self.num_layers
#             gap = self.group_sz**butterfly_layer_index
#             if gap*self.group_sz > self.inplanes:
#                 gap = int(np.ceil(self.inplanes/self.group_sz))
            
            
            
            x = x.view(B, -1, self.group_sz, gap, H, W).transpose(2, 3).contiguous().view(B, -1, H, W)
            x = fn(x)
            _, _, H, W = x.shape
            x = x.view(B, -1, gap, self.group_sz, H, W).transpose(2, 3).contiguous().view(B, -1, H, W)

#         x = x.view(B, C, H, W)
        return x
        

class CifarResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, planes=16, group_sizes=None):
        super(CifarResNet, self).__init__()
        global conv3x3, conv1x1
        
        ### buffer for last used planes in conv-res-blocks
        self.inplanes = planes
        self.conv1 = conv3x3(3, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        if group_sizes is None:
            group_sizes = [-1, -1, -1]
                
        self.layer1 = self._make_layer(block, planes, layers[0], group_sz=group_sizes[0])
        self.layer2 = self._make_layer(block, planes*2, layers[1], stride=2, group_sz=group_sizes[1])
        self.layer3 = self._make_layer(block, planes*4, layers[2], stride=2, group_sz=group_sizes[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(planes*4 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, group_sz=-1):
        downsample = None
        if group_sz <= 0:
            groups = 1
        else:
            groups = self.inplanes//group_sz
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, groups=groups),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=groups))
        
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups))

        return SequentialMixer(layers, self.inplanes, group_sz)

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
    model = CifarResNet(BasicBlock, [3, 3, 3], group_sizes=[4, 8, 8], **kwargs)
    return model

def cifar_resnet23(**kwargs):
    model = CifarResNet(BasicBlock, [4, 4, 4], group_sizes=[4, 8, 8], **kwargs)
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