import torch
import torch.nn as nn
import numpy as np

#### these are compiled from cuda kernels
import bmm2x2_cuda
import bilinear2x2_cuda


############################################################################
############################################################################

## Cuda -bmm2x2
class BMM2x2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        outputs = bmm2x2_cuda.forward(inputs, weights)
        ctx.save_for_backward(inputs, weights)
        return outputs[0]
    
    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        del_input, del_weights = bmm2x2_cuda.backward(
            inputs, 
            weights, 
            grad_output)
    
        return del_input, del_weights

class PairLinear(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        assert input_dim%2 == 0, "Input dim must be even number"
        self.weight = torch.eye(2).unsqueeze(0).repeat_interleave(input_dim//2, dim=0)
        self.weight = nn.Parameter(self.weight)
        self.bmmfunc = BMM2x2Function()
        
    def forward(self, x):
        bs, dim = x.shape[0], x.shape[1]
        x = x.view(bs, -1, 2)
        x = BMM2x2Function.apply(x, self.weight)
        x = x.view(bs, -1)
        return x
    
    def __repr__(self):
        t = self.weight.shape[0]*2
        S = f'PairLinear: [{t} -> {t}]'
        return S

### BMM 2x1
class BMM2x1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        outputs = bmm2x2_cuda.forward_2x1(inputs, weights)
        ctx.save_for_backward(inputs, weights)
        return outputs[0]
    
    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        del_input, del_weights = bmm2x2_cuda.backward_2x1(
            inputs, 
            weights, 
            grad_output)
    
        return del_input, del_weights

class PairLinearHalve(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        assert input_dim%2 == 0, "Input dim must be even number"
        self.weight = torch.Tensor([0.5, 0.5]).unsqueeze(0).repeat_interleave(input_dim//2, dim=0)
        self.weight = nn.Parameter(self.weight)
        
    def forward(self, x):
        bs, dim = x.shape[0], x.shape[1]
        x = x.view(bs, -1, 2)
        x = BMM2x1Function.apply(x, self.weight)
        x = x.view(bs, -1)
        return x
    
    def __repr__(self):
        t = self.weight.shape[0]
        S = f'PairLinearHalve: [{t*2} -> {t}]'
        return S
    
    
############################################################################
############################################################################


class BiasLayer(nn.Module):
    def __init__(self, dim, init_val=0):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim)*init_val)
        
    def forward(self, x):
        return x+self.bias
    
    def __repr__(self):
        S = f'BiasLayer: [{self.bias.shape[0]}]'
        return S

class DimensionSelector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        assert output_dim > input_dim, "Slection does not select all inputs"
        remain = output_dim-input_dim
        
        scale = int(np.ceil(output_dim/input_dim)-1)
#         self.indices = torch.randperm(input_dim*scale)[:remain]%input_dim

        self.indices = torch.LongTensor([])
        for i in range(scale):
            c = min(input_dim, remain-len(self.indices))
            t = torch.randperm(input_dim)[:c]
            self.indices = torch.cat([self.indices, t])
            
        
    def forward(self, x):
        ## x.shape = [batch_size, input_dim]
        return torch.cat([x, x[:, self.indices]], dim=1)
    
    def __repr__(self):
        S = f'DimensionSelector: [+={self.indices.shape[0]}]'
        return S
    
############################################################################
############################################################################

class PairLinear_MixerBlock(nn.Module):
    
    '''
    Handle any input - output size;
    
    Operations -> Select, NxN mix, Halve
    
    -Edge cases:
    1) 8-8 -> NxN mixing for log(N) times
    2) 8-10 -> Select(16) + 16x16 + Select(20) + Halve
    3) 8-6 -> 8x8 + Select(12) + Halve
    4) 8-32 -> Select(32) + 32x32
    5) 8-3 -> 8x8 + Halve + 4-Select(6) + Halve
    
    '''
    
    def __init__(self, input_dim, output_dim, bias=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.selector = None
        self.pairwise_mixing = []
        self.reducer = []
        
        mix_dim = 2**int(np.ceil(np.log2(max(input_dim, output_dim))))
        
        #########################################################
        ### Find out if first selection is required or Not !
        if self.input_dim != mix_dim:
            ## Input dimension is not power of 2; requires selector to project to mixing dimension
            L = DimensionSelector(input_dim, mix_dim)
            self.selector = L
        else:
            self.selector = nn.Identity()
        
        ### Now perform NxN mixing 
        num_layers = int(np.ceil(np.log2(mix_dim)))
        for i in range(num_layers):
            net = PairLinear(mix_dim)
            self.pairwise_mixing.append(net)
        self.pairwise_mixing = nn.ModuleList(self.pairwise_mixing)
        
        ### Now for reducer if any
        num_halve = int(np.ceil(np.log2(mix_dim/output_dim)))
        final_expand = output_dim*(2**num_halve)
        if final_expand != mix_dim:
            L = DimensionSelector(mix_dim, final_expand)
            self.reducer.append(L)
        for i in range(num_halve):
            L = PairLinearHalve(final_expand//(2**i))
            self.reducer.append(L)
            
        if len(self.reducer) == 0:
            self.reducer = nn.Identity()
            if bias:
                self.reducer = BiasLayer(output_dim, 0)
        else:
            self.reducer = nn.Sequential(*self.reducer, BiasLayer(mix_dim, 0))
        
        pass
    
    def forward(self, x):
        '''
        x: shape-> [batch_size, input_dim]
        '''
        bs = x.shape[0]
        
        x = self.selector(x)
        
        y = x
        for i, fn in enumerate(self.pairwise_mixing):
            y = y.view(-1,2,2**i).permute(0, 2,1).contiguous().view(bs, -1)
            y = fn(y)
            y = y.view(-1,2**i,2).permute(0, 2,1).contiguous()

        y = y.view(bs, -1)
#         y = x + y ## this is residual addition... remove if only want feed forward
        y = self.reducer(y)
        return y


############################################################################
############################################################################

class BlockWeight(nn.Module):
    def __init__(self, input_dim, block_dim):
        super().__init__()
        self.block_dim = block_dim
        
        assert input_dim%block_dim == 0, "Input dim must be even number"
        self.weight = torch.eye(block_dim).unsqueeze(0).repeat_interleave(input_dim//block_dim, dim=0)
        self.weight = nn.Parameter(self.weight)
        
    def forward(self, x):
        bs, dim = x.shape[0], x.shape[1]
        x = x.view(bs, -1, self.block_dim).transpose(0,1)
        x = torch.bmm(x, self.weight)
        x = x.transpose(1,0).reshape(bs, -1)
        return x
    
    
class BlockLinear_MixerBlock(nn.Module):
    
    def __init__(self, input_dim, block_dim):
        super().__init__()
        
        assert input_dim%block_dim == 0, "Input dim must be even number"
        self.input_dim = input_dim
        self.block_dim = block_dim
        
        def log_base(a, base):
            return np.log(a) / np.log(base)
        
        num_layers = int(np.ceil(log_base(input_dim, base=block_dim)))
            
        self.facto_nets = []
        for i in range(num_layers):
            net = BlockWeight(self.input_dim, block_dim)
            self.facto_nets.append(net)
            
        self.facto_nets = nn.ModuleList(self.facto_nets)
            
    def forward(self, x):
        bs = x.shape[0]
        y = x
        for i, fn in enumerate(self.facto_nets):
            y = y.view(-1,self.block_dim,self.block_dim**i).permute(0, 2, 1).contiguous().view(bs, -1)
            y = fn(y)
            y = y.view(-1,self.block_dim**i,self.block_dim).permute(0, 2, 1).contiguous()

        y = y.view(bs, -1)
        return y


############################################################################
############################################################################