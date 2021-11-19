import torch
import torch.nn as nn
import torch.nn.functional as F

from flows import Flow
from coupling_flows import DimensionMixer

class DownScale(Flow):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def _forward_yes_logDetJ(self, x):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = x.shape[0], x.shape[1], x.shape[2] // bl, x.shape[3] // bl
        z = x.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)
        log_det = 0
        return z, log_det

    def _forward_no_logDetJ(self, x):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = x.shape[0], x.shape[1], x.shape[2] // bl, x.shape[3] // bl
        z = x.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)
        return z
    
    def _inverse_yes_logDetJ(self, z):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = z.shape[0], z.shape[1] // bl_sq, z.shape[2], z.shape[3]
        x = z.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl)
        log_det = 0
        return x, log_det

    def _inverse_no_logDetJ(self, z):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = z.shape[0], z.shape[1] // bl_sq, z.shape[2], z.shape[3]
        x = z.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl)
        return x

class UpScale(Flow):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def _forward_yes_logDetJ(self, x):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = x.shape[0], x.shape[1] // bl_sq, x.shape[2], x.shape[3]
        z = x.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl)
        log_det = 0
        return z, log_det

    def _forward_no_logDetJ(self, x):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, new_d, h, w = x.shape[0], x.shape[1] // bl_sq, x.shape[2], x.shape[3]
        z = x.reshape(bs, bl, bl, new_d, h, w).permute(0, 3, 4, 1, 5, 2).reshape(bs, new_d, h * bl, w * bl)
        return z
    
    def _inverse_yes_logDetJ(self, z):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = z.shape[0], z.shape[1], z.shape[2] // bl, z.shape[3] // bl
        x = z.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)
        log_det = 0
        return x, log_det

    def _inverse_no_logDetJ(self, z):
        bl, bl_sq = self.block_size, self.block_size_sq
        bs, d, new_h, new_w = z.shape[0], z.shape[1], z.shape[2] // bl, z.shape[3] // bl
        x = z.reshape(bs, d, new_h, bl, new_w, bl).permute(0, 3, 5, 1, 2, 4).reshape(bs, d * bl_sq, new_h, new_w)
        return x


class ConvNetGenerator():

    def __init__(self, in_channel, channels:list, kernels=3, batch_norm=True, activation=nn.ReLU, dropout_p=0):
        self.in_channel = in_channel
        self.channels = channels
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout_p = dropout_p

        if isinstance(kernels, int):
            self.kernels = [kernels for _ in range(len(channels)+1)]
        else:
            assert len(channels)+1 == len(kernels), "The length of kernels must be one greater than number of channels"
            self.kernels = kernels

        self.paddings = [(kernel-1)//2 for kernel in self.kernels]

    def generate(self):
        layers = []
        channels = [int(self.in_channel)]+\
                    self.channels + [int(self.in_channel)]

        for i in range(len(channels)-1):
            conv = nn.Conv2d(channels[i], channels[i+1], self.kernels[i], 
                                    padding=self.paddings[i], bias=not self.batch_norm)
            layers.append(conv)
            if self.batch_norm:
                bn = nn.BatchNorm2d(channels[i+1])
                layers.append(bn)
            if self.dropout_p > 0:
                dropout = nn.Dropout(p=self.dropout_p)
                layers.append(dropout)
            layers.append(self.activation())
            
        layers = layers[:-1]
        layers = nn.Sequential(*layers)
        return layers





class iRevNet(Flow):
    def __init__(self,sample_dim, func_generator, sample_ratio=1):
        super().__init__()
        self.dim = func_generator.in_channels
        self.sample_ratio = sample_ratio

        if self.sample_ratio > 1:
            self.layers.append(UpScale(int(self.sample_ratio)))
        elif self.sample_ratio < 1:
            self.layers.append(DownScale(int(1/self.sample_ratio)))

        # self.parity = parity
        if sample_dim == 0:
            self.input_index = torch.LongTensor(list(range(self.dim))[::2])
            self.modif_index = torch.LongTensor(list(range(self.dim))[1::2])
        elif sample_dim == 1: ## opposite
            self.input_index = torch.LongTensor(list(range(self.dim))[1::2])
            self.modif_index = torch.LongTensor(list(range(self.dim))[::2])
        elif isinstance(sample_dim, DimensionMixer):
            self.input_index, self.modif_index = sample_dim.get_input_and_modify_index()

        self.t_cond = func_generator.generate()
        
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


if __name__ == '__main__':
    cg = ConvNetGenerator(3, [16, 32])
    cg.generate()