import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

##################################################################

class ConvexNN(nn.Module):
    
    def __init__(self, dims:list, actf=nn.LeakyReLU):
        super().__init__()
        assert len(dims)>1
        self.dims = dims
        layers = []
        skip_layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i>0:
                skip_layers.append(nn.Linear(dims[0], dims[i+1]))
                skip_layers[-1].weight.data *= 0.1
                layers[-1].weight.data *= 0.1
            if i<len(dims)-2:
                layers.append(actf())
            
        self.layers = nn.ModuleList(layers)
        self.skip_layers = nn.ModuleList(skip_layers)
        
    def forward(self,x):
        h = x
        for i in range(len(self.dims)-1):
            self.layers[i*2].weight.data = torch.abs(self.layers[i*2].weight.data)
            h = self.layers[i*2](h)
            if i>0:
                h += self.skip_layers[i-1](x)
            if i<len(self.dims)-2:
                h = self.layers[i*2+1](h)
        return h