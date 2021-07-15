import torch
import torch.nn as nn
from .flows import Flow


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

# ------------------------------------------------------------------------

class AffineHalfFlow(Flow):
    """
    As seen in RealNVP, affine autoregressive flow (z = x * exp(s) + t), where half of the 
    dimensions in x are linearly scaled/transfromed as a function of the other half.
    Which half is which is determined by the parity bit.
    - RealNVP both scales and shifts (default)
    - NICE only shifts
    """
    def __init__(self, dim, sample_dim, func_generator, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        
        # self.parity = parity
        if sample_dim == 0:
            self.input_index = torch.LongTensor(list(range(self.dim))[::2])
            self.modif_index = torch.LongTensor(list(range(self.dim))[1::2])
        elif sample_dim == 1: ## opposite
            self.input_index = torch.LongTensor(list(range(self.dim))[1::2])
            self.modif_index = torch.LongTensor(list(range(self.dim))[::2])
        elif isinstance(sample_dim, DimensionMixer):
            self.input_index, self.modif_index = sample_dim.get_input_and_modify_index()

        self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim // 2)
        if scale:
            self.s_cond = func_generator.generate(self.dim // 2, self.dim // 2)
        if shift:
            self.t_cond = func_generator.generate(self.dim // 2, self.dim // 2)
        
    def _forward_yes_logDetJ(self, x):
        # x0, x1 = x[:,::2], x[:,1::2]
        # if self.parity:
        #     x0, x1 = x1, x0
        x0, x1 = x[:,self.input_index], x[:,self.modif_index]
        
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        # z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        # if self.parity:
        #     z0, z1 = z1, z0
        # z = torch.cat([z0, z1], dim=1)
        z = torch.empty_like(x)
        z[:, self.input_index] = x0
        z[:, self.modif_index] = z1

        log_det = torch.sum(s, dim=1)
        return z, log_det

    def _forward_no_logDetJ(self, x):
        # x0, x1 = x[:,::2], x[:,1::2]
        # if self.parity:
        #     x0, x1 = x1, x0
        x0, x1 = x[:,self.input_index], x[:,self.modif_index]
        
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        # z0 = x0 # untouched half
        z1 = torch.exp(s) * x1 + t # transform this half as a function of the other
        # if self.parity:
        #     z0, z1 = z1, z0
        # z = torch.cat([z0, z1], dim=1)

        z = torch.empty_like(x)
        z[:, self.input_index] = x0
        z[:, self.modif_index] = z1

        return z
    
    def _inverse_yes_logDetJ(self, z):
        # z0, z1 = z[:,::2], z[:,1::2]
        # if self.parity:
        #     z0, z1 = z1, z0
        z0, z1 = z[:,self.input_index], z[:,self.modif_index]
        
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        # x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        # if self.parity:
        #     x0, x1 = x1, x0
        # x = torch.cat([x0, x1], dim=1)

        x = torch.empty_like(z)
        x[:, self.input_index] = z0
        x[:, self.modif_index] = x1

        log_det = torch.sum(-s, dim=1)
        return x, log_det

    def _inverse_no_logDetJ(self, z):
        # z0, z1 = z[:,::2], z[:,1::2]
        # if self.parity:
        #     z0, z1 = z1, z0

        z0, z1 = z[:,self.input_index], z[:,self.modif_index]

        s = self.s_cond(z0)
        t = self.t_cond(z0)
        # x0 = z0 # this was the same
        x1 = (z1 - t) * torch.exp(-s) # reverse the transform on this half
        # if self.parity:
        #     x0, x1 = x1, x0
        # x = torch.cat([x0, x1], dim=1)
        x = torch.empty_like(z)
        x[:, self.input_index] = z0
        x[:, self.modif_index] = x1

        return x