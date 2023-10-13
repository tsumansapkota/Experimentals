import torch
import torch.nn as nn
import numpy as np


############################################################################
############################################################################

class BlockWeight(nn.Module):
    def __init__(self, input_dim, block_dim):
        super().__init__()
        self.block_dim = block_dim
        
        assert input_dim%block_dim == 0, "Input dim must be even number"
        self.weight = torch.eye(block_dim).unsqueeze(0).repeat_interleave(input_dim//block_dim, dim=0)
        self.weight = nn.Parameter(self.weight)
        self.reset_parameters()
        
    def forward(self, x):
        bs, dim = x.shape[0], x.shape[1]
        x = x.view(bs, -1, self.block_dim).transpose(0,1)
        x = torch.bmm(x, self.weight)
        x = x.transpose(1,0).reshape(bs, -1)
        return x
    
    def __repr__(self):
        S = f'BlockWeight: {list(self.weight.shape)}'
        return S
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        pass
    
    
class BlockLinear_MixerBlock(nn.Module):
    
    def __init__(self, input_dim, block_dim, bias=False):
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

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, input_dim))
            
    def forward(self, x):
        bs = x.shape[0]
        y = x
        for i, fn in enumerate(self.facto_nets):
            y = y.view(-1, self.block_dim, self.block_dim**i).permute(0, 2, 1).contiguous().view(bs, -1)
            y = fn(y)
            y = y.view(-1, self.block_dim**i, self.block_dim).permute(0, 2, 1).contiguous()

        y = y.view(bs, -1)
        if self.bias is not None:
            y = y + self.bias
        return y

############################################################################
############################################################################
#### NEXT: BLOCKS ARE NON-LINEAR
############################################################################
############################################################################
    
    
    
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
        self.reset_parameters()    
        
        
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
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        pass

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
#             print(l.weight.shape)
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
        for i in range(num_layers):
            net = BlockMLP(self.input_dim, block_layer_dims, actf)
            self.facto_nets.append(net)
            
        self.facto_nets = nn.ModuleList(self.facto_nets)
            
    def forward(self, x):
        bs = x.shape[0]
        y = x
        for i, fn in enumerate(self.facto_nets):
            y = y.view(-1, self.block_dim, self.block_dim**i).permute(0, 2, 1).contiguous().view(bs, -1)
            y = fn(y)
            y = y.view(-1, self.block_dim**i, self.block_dim).permute(0, 2, 1).contiguous()

        y = y.view(bs, -1)
        return y

