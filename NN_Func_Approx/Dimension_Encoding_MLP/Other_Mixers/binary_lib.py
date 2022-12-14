import torch
import torch.nn as nn
import numpy as np


#############################################################################
#############################################################################

class ResMlpBlock(nn.Module):
    
    def __init__(self, input_dim, hidden_layers_ratio=[2], actf=nn.GELU):
        super().__init__()
        self.input_dim = input_dim
        #### convert hidden layers ratio to list if integer is inputted
        if isinstance(hidden_layers_ratio, int):
            hidden_layers_ratio = [hidden_layers_ratio]
            
        self.hlr = [1]+hidden_layers_ratio+[1]
        
        self.mlp = []
        ### for 1 hidden layer, we iterate 2 times
        for h in range(len(self.hlr)-1):
            i, o = int(self.hlr[h]*self.input_dim),\
                    int(self.hlr[h+1]*self.input_dim)
            self.mlp.append(nn.Linear(i, o))
            self.mlp.append(actf())
        self.mlp = self.mlp[:-1]
        
        self.mlp = nn.Sequential(*self.mlp)
        
    def forward(self, x):
        return self.mlp(x)+x
    
    
#############################################################################
#############################################################################
    
    

############################################################################
######### FOR BLOCK MLP
############################################################################

class BlockLinear(nn.Module):
    def __init__(self, num_blocks, input_block_dim, output_block_dim, bias=True):
        super().__init__()
        self.weight = torch.randn(num_blocks, input_block_dim, output_block_dim)
        
        self.weight = nn.Parameter(self.weight)
        
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.weight.shape[0], 1, output_block_dim))
        
    def forward(self, x):
#         nblocks, bs, dim = x.shape[0], x.shape[1], x.shape[2]
#         print(x.shape)
        x = torch.bmm(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x
    
    def __repr__(self):
        S = f'BlockLinear: {list(self.weight.shape)}'
        return S
    
############################################################################
############################################################################

class BlockMLP(nn.Module):
    def __init__(self, input_dim, layer_dims, actf=nn.ELU):
        super().__init__()
        self.block_dim = layer_dims[0]
        
        assert input_dim%self.block_dim == 0, "Input dim must be even number"
        ### Create a block MLP
        self.mlp = []
        n_blocks = input_dim//layer_dims[0]
        for i in range(len(layer_dims)-1):
            l = BlockLinear(n_blocks, layer_dims[i], layer_dims[i+1])
            a = actf()
            self.mlp.append(l)
            self.mlp.append(a)
            
        self.mlp = self.mlp[:-1]
        self.mlp = nn.Sequential(*self.mlp)
#         self.ln = nn.LayerNorm(self.block_dim)
        
    def forward(self, x):
        bs, dim = x.shape[0], x.shape[1]
        x = x.view(bs, -1, self.block_dim).transpose(0,1)
#         x = self.mlp(self.ln(x)) + x
        x = self.mlp(x) + x
        x = x.transpose(1,0).reshape(bs, -1)
        return x
    
############################################################################
############################################################################

class BlockMLP_MixerBlock(nn.Module):
    
    def __init__(self, input_dim, block_dim, hidden_layers_ratio=[2], actf=nn.ELU):
        super().__init__()
        
        assert input_dim%block_dim == 0, "Input dim must be even number"
        self.input_dim = input_dim
        self.block_dim = block_dim
        
        def log_base(a, base):
            return np.log(a) / np.log(base)
        
        num_layers = int(np.ceil(log_base(input_dim, base=block_dim)))
        hidden_layers_ratio = [1] + hidden_layers_ratio + [1]
        
        block_layer_dims = [int(a*block_dim) for a in hidden_layers_ratio]
        self.facto_nets = []
        self.gaps = []
        for i in range(num_layers):
            net = BlockMLP(self.input_dim, block_layer_dims, actf)
            self.facto_nets.append(net)
            
            gap = self.block_dim**i
            if gap*self.block_dim <= self.input_dim:
                self.gaps.append(gap)
            else:
                self.gaps.append(int(np.ceil(self.input_dim/self.block_dim)))
            
        self.facto_nets = nn.ModuleList(self.facto_nets)
            
    def forward(self, x):
        bs = x.shape[0]
        y = x
        for i, fn in enumerate(self.facto_nets):
            gap = self.gaps[i]
            y = y.view(-1, self.block_dim, gap).transpose(2, 1).contiguous().view(bs, -1)
            y = fn(y)
            y = y.view(-1, gap, self.block_dim).transpose(2, 1)

        y = y.contiguous().view(bs, -1)
        return y

#############################################################################
#############################################################################



#############################################################################
#############################################################################


    
    
#############################################################################
#############################################################################





#############################################################################
#############################################################################





