import torch
import torch.nn as nn
import numpy as np

#### these are compiled from cuda kernels
import bilinear2x2_cuda
import bmm2x2_cuda

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
    
    def __repr__(self):
        S = f'BlockWeight: {list(self.weight.shape)}'
        return S
    
    
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
            y = y.view(-1, self.block_dim, self.block_dim**i).permute(0, 2, 1).contiguous().view(bs, -1)
            y = fn(y)
            y = y.view(-1, self.block_dim**i, self.block_dim).permute(0, 2, 1).contiguous()

        y = y.view(bs, -1)
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
            y = y.view(-1, self.block_dim**i, self.block_dim).permute(0, 2, 1)

        y = y.contiguous().view(bs, -1)
        return y

############################################################################
### FOR PAIR BILINEAR
############################################################################

class BiLinear2x2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        outputs = bilinear2x2_cuda.forward(inputs, weights)
        ctx.save_for_backward(inputs, weights)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        del_input, del_weights = bilinear2x2_cuda.backward(
            inputs, 
            weights, 
            grad_output)
    
        return del_input, del_weights
    

class PairBilinear(nn.Module):
    def __init__(self, dim, grid_width):
        super().__init__()
        num_pairs = dim // 2
        along_row = torch.linspace(0, 1, grid_width).reshape(1, -1).t()
        along_col = torch.linspace(0, 1, grid_width).reshape(-1, 1).t()
        
        self.pairW = torch.eye(2).unsqueeze(0).repeat_interleave(num_pairs, dim=0)
        self.pairW = nn.Parameter(self.pairW)
    
        self.Y = torch.stack([along_row+along_col*0, along_row*0+along_col])
        self.Y = torch.repeat_interleave(self.Y.unsqueeze(0), num_pairs, dim=0)
        self.Y = nn.Parameter(self.Y)
        
        del_x = 1/(grid_width-1) ## lipschitz constraint if multiplied by 1
        slope = 2
        self.dy = slope*del_x
        
        
    def forward(self, x):
        bs = x.shape[0]
        
        x = x.view(bs, -1, 2)
#         x = BMM2x2Function.apply(x, self.pairW)
        ####################################################
        
        ### Constrain the bilinear to have specific slope, < 5
        if self.training:
            init0 = self.Y.data[:,:,:,:1]
            init1 = self.Y.data[:,:,:1,:]
            a0 = torch.diff(self.Y.data, dim=-1).clamp(-self.dy, self.dy)
            a1 = torch.diff(self.Y.data, dim=-2).clamp(-self.dy, self.dy)
            a0 = torch.cat([init0, a0], dim=-1)
            a1 = torch.cat([init1, a1], dim=-2)
            f0 = torch.cumsum(a0, dim=-1)
            f1 = torch.cumsum(a1, dim=-2)
            f = (f0+f1)/2
            self.Y.data = f
        
        ####################################################
        x = BiLinear2x2Function.apply(x, self.Y)
        x = x.view(bs, -1)
        return x
    
    def __repr__(self):
        t = self.Y.shape[0]*2
        u = self.Y.shape[2]
        S = f'PairBilinear: [{t} -> {t}] (grid: {u})'
        return S


############################################################################
############################################################################

class BiLinear2x1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        outputs = bilinear2x2_cuda.forward_2x1(inputs, weights)
        ctx.save_for_backward(inputs, weights)
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        del_input, del_weights = bilinear2x2_cuda.backward_2x1(
            inputs, 
            weights, 
            grad_output)
    
        return del_input, del_weights

class PairBilinearHalve(nn.Module):
    def __init__(self, dim, grid_width):
        super().__init__()
        num_pairs = dim // 2
        
        self.pairW = torch.eye(2).unsqueeze(0).repeat_interleave(num_pairs, dim=0)
        self.pairW = nn.Parameter(self.pairW)

        along_row = torch.linspace(0, 1, grid_width).reshape(1, -1).t()
        along_col = torch.linspace(0, 1, grid_width).reshape(-1, 1).t()
        
        self.Y = torch.stack([along_row+along_col*0, along_row*0+along_col]).mean(dim=0)
        self.Y = torch.repeat_interleave(self.Y.unsqueeze(0), num_pairs, dim=0)
        self.Y = nn.Parameter(self.Y)
        
    
    def forward(self, x):
        bs = x.shape[0]
        
        x = x.view(bs, -1, 2)
        x = BMM2x2Function.apply(x, self.pairW)
        ####################################################
        x = BiLinear2x1Function.apply(x, self.Y)
        x = x.view(bs, -1)
        return x
    
    def __repr__(self):
        t = self.pairW.shape[0]
        u = self.Y.shape[2]
        S = f'PairLinear: [{t*2} -> {t}] (grid: {u})'
        return S

############################################################################
############################################################################

class PairBilinear_MixerBlock(nn.Module):
    
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
    
    def __init__(self, input_dim, output_dim, grid_width, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_width = grid_width
        
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
            if bias:
                self.selector = nn.Sequential(L, BiasLayer(mix_dim, 0.5))
        else:
            self.selector = nn.Identity()
            if bias:
                self.selector = BiasLayer(mix_dim, 0.5)
        
        ### Now perform NxN mixing 
        num_layers = int(np.ceil(np.log2(mix_dim)))
        for i in range(num_layers):
            net = PairBilinear(mix_dim, grid_width)
            net.Y.data *= 0.0
#             net.pairW.data *= 0.5
            self.pairwise_mixing.append(net)
        self.pairwise_mixing = nn.ModuleList(self.pairwise_mixing)
        
        ### Now for reducer if any
        num_halve = int(np.ceil(np.log2(mix_dim/output_dim)))
        final_expand = output_dim*(2**num_halve)
        if final_expand != mix_dim:
            L = DimensionSelector(mix_dim, final_expand)
            self.reducer.append(L)
        for i in range(num_halve):
#             L = PairBilinearHalve(final_expand//(2**i), grid_width)
            L = PairLinearHalve(final_expand//(2**i))
            self.reducer.append(L)
            
        if len(self.reducer) == 0:
            self.reducer = nn.Identity()
        else:
            self.reducer = nn.Sequential(*self.reducer)
        
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
            y = fn(y)+y
            y = y.view(-1,2**i,2).permute(0, 2,1)

        y = y.contiguous().view(bs, -1)
#         y = x + y ## this is residual addition... remove if only want feed forward
        y = self.reducer(y)
        return y

############################################################################
############################################################################

