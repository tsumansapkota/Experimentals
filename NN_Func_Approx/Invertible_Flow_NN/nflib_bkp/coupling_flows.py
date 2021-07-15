import torch
from torch import functional
import torch.nn as nn
from .basic_flows import Flow
from typing import Union


class FuncGenerator():

    def generate():
        raise NotImplementedError

class MLP_Generator(FuncGenerator):

    def __init__(self, hidden_dims=[], activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation

    def generate(self, input_dim, output_dim):
        dims = [input_dim, *self.hidden_dims]
        layers = []
        for i in range(len(dims)-1):
            a = nn.Linear(dims[i], dims[i+1])
            b = self.activation
            layers += [a, b]
        fl = nn.Linear(dims[-1], output_dim)
        fl.weight.data *= 0.
        layers.append(fl)
        return nn.Sequential(*layers)


class DimensionMixer():

    def __init__(self, dim):
        self.dim = dim
        self.select_count = torch.zeros(dim)
    
    def get_input_and_modify_index(self):
        ### sorts in ascending order
        priority = torch.argsort(self.select_count + torch.randn(self.dim)*0.001)

        ### low value will be input
        inp_indx, mod_indx = priority[:self.dim//2], priority[self.dim//2:]
        
        ## in next priority, high value will make inp_indx to be mod_indx and viceversa
        self.select_count[inp_indx] += 1 
        return inp_indx, mod_indx



#### major flaw in dimension stitching in forward and inverse
# !!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!
class CouplingFlow(Flow):

    def __init__(self, dim, func_generator:FuncGenerator, dim_sample:Union[None, DimensionMixer]=None, scale=True, shift=True):
        super().__init__()
        
        assert scale or shift , "Coupling flow must either scale or shift or both"
        assert isinstance(dim_sample, DimensionMixer) or dim_sample == 0 or dim_sample == 1

        self.dim = dim
        self.func_generator = func_generator

        if dim_sample == 0:
            self.input_index = torch.LongTensor(list(range(self.dim))[::2])
            self.modif_index = torch.LongTensor(list(range(self.dim))[1::2])
        elif dim_sample == 1: ## opposite
            self.modif_index = torch.LongTensor(list(range(self.dim))[::2])
            self.input_index = torch.LongTensor(list(range(self.dim))[1::2])
        elif isinstance(dim_sample, DimensionMixer):
            self.input_index, self.modif_index = dim_sample.get_input_and_modify_index()

        # self.shift, self.scale = shift, scale

        self.scale_func = lambda x: torch.zeros(x.size(0), self.dim // 2)
        self.shift_func = lambda x: torch.zeros(x.size(0), self.dim // 2)
        if scale:
            self.scale_func = func_generator.generate(self.dim//2, self.dim//2)
        if shift:
            self.shift_func = func_generator.generate(self.dim//2, self.dim//2)


    def forward(self, x, logdetJ=False):
        x0, x1 = x[:,self.input_index], x[:,self.modif_index]
        s = self.scale_func(x0)
        t = self.shift_func(x0)
        y1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        y = torch.cat([x0, y1], dim=1)
        if logdetJ:
            log_det = torch.sum(s, dim=1)
            return y, log_det
        else:
            return y
    
    def inverse(self, y, logdetJ=False):
        y0, y1 = y[:,self.input_index], y[:,self.modif_index]
        s = self.scale_func(y0)
        t = self.shift_func(y0)
        x1 = (y1 - t) * torch.exp(-s) # reverse the transform on this half
        x = torch.cat([y0, x1], dim=1)
        if logdetJ:
            log_det = torch.sum(s, dim=1)
            return x, log_det
        else:
            return x


        
