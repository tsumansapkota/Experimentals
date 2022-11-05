import torch
import torch.nn as nn
import numpy as np


#############################################################################
#############################################################################

class ResMlpBlock(nn.Module):
    
    def __init__(self, input_dim, hidden_layers_ratio=[2], dropout=0.0, actf=nn.GELU):
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
            if dropout > 0:
                self.mlp.append(nn.Dropout(dropout))
            
        self.mlp = self.mlp[:-1] ## remove actf from final layer
        if dropout > 0:
            self.mlp = self.mlp[:-1] ## remove dropout from final layer
        
        self.mlp[-1].weight.data *= 0. ## initialize to have no effect on output
        self.mlp = nn.Sequential(*self.mlp)
        
    def forward(self, x):
        return self.mlp(x)+x


#############################################################################
#############################################################################


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
#         self.pe = pe

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


#############################################################################
#############################################################################
############### CUSTOM FOR SPARSE TRANSFORMER #########################

    
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
        print(x.shape, 'before linear')

        x = torch.bmm(x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x
    
    def __repr__(self):
        S = f'BlockLinear: {list(self.weight.shape)}'
        return S
    
############################################################################
############################################################################

class BlockResMLP(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout=0, actf=nn.ELU):
        super().__init__()
        self.block_dim = layer_dims[0]
        
        assert input_dim%self.block_dim == 0, "Input dim must be even number"
        ### Create a block MLP
        self.mlp = []
        n_blocks = input_dim//layer_dims[0]
        for i in range(len(layer_dims)-1):
            l = BlockLinear(n_blocks, layer_dims[i], layer_dims[i+1])
            a = actf()
            if dropout > 0:
                d = nn.Dropout(dropout)
            self.mlp.append(l)
            self.mlp.append(a)
            
        self.mlp = self.mlp[:-1]
        if dropout > 0 : 
            self.mlp = self.mlp[:-1]
            
        self.mlp = nn.Sequential(*self.mlp)
        
    def forward(self, x):
        bs, dim = x.shape[0], x.shape[1]
        print(x.shape, 'before mlp')
        
        x = x.view(bs, -1, self.block_dim).transpose(0,1)
        x = self.mlp(x) + x
        x = x.transpose(1,0).reshape(bs, -1)
        return x
    
############################################################################
############################################################################

class BlockResMLP_MixerBlock(nn.Module):
    
    def __init__(self, input_dim, block_dim, hidden_layers_ratio=[2], dropout=0, actf=nn.ELU):
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
            net = BlockResMLP(self.input_dim, block_layer_dims, dropout, actf)
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
            print(y.shape, 'before block mlp')
            y = y.view(-1, self.block_dim, gap).transpose(2, 1).contiguous().view(bs, -1)
            y = fn(y)
            y = y.view(-1, gap, self.block_dim).transpose(2, 1)

        y = y.contiguous().view(bs, -1)
        return y
    
############################################################################
############################################################################


class BlockMLP(nn.Module):
    def __init__(self, input_dim, layer_dims, dropout=0, actf=nn.ELU):
        super().__init__()
        self.block_dim = layer_dims[0]
        
        assert input_dim%self.block_dim == 0, "Input dim must be even number"
        ### Create a block MLP
        self.mlp = []
        n_blocks = input_dim//layer_dims[0]
        for i in range(len(layer_dims)-1):
            l = BlockLinear(n_blocks, layer_dims[i], layer_dims[i+1])
            a = actf()
            if dropout > 0:
                d = nn.Dropout(dropout)
            self.mlp.append(l)
            self.mlp.append(a)
            
        self.mlp = self.mlp[:-1]
        if dropout > 0 : 
            self.mlp = self.mlp[:-1]
            
        self.mlp = nn.Sequential(*self.mlp)
        
    def forward(self, x):
        bs, dim = x.shape[0], x.shape[1]
        x = x.view(bs, -1, self.block_dim).transpose(0,1)
        x = self.mlp(x)
        x = x.transpose(1,0).reshape(bs, -1)
        return x
    
############################################################################
############################################################################

class BlockMLP_MixerBlock(nn.Module):
    
    def __init__(self, input_dim, block_dim, hidden_layers_ratio=[2], dropout=0, actf=nn.ELU):
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
            net = BlockMLP(self.input_dim, block_layer_dims, dropout, actf)
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

        y = y.contiguous().view(bs, -1) + x
        return y

    
    
#############################################################################
#############################################################################
 
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, actf=nn.GELU, embed_block_size:int=None):
        super(TransformerBlock, self).__init__()
        
        self.attention = nn.MultiheadAttention(embed_size, heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        if embed_block_size is None or embed_block_size == embed_size:
            hid = int(forward_expansion * embed_size)
            self.feed_forward = ResMlpBlock(embed_size, [forward_expansion], dropout, actf)
        else:
            self.feed_forward = BlockResMLP_MixerBlock(embed_size, embed_block_size, [forward_expansion], dropout, actf)
        pass

    def forward(self, query):
        q = self.norm1(query)
        query = self.attention(q, q, q, need_weights=False)[0] + query
                
        shape = query.shape
        query = query.view(-1, shape[-1])
        
        query = self.feed_forward(self.norm2(query)) + query

        return query.view(*shape)
    
#############################################################################
#############################################################################



class Mixer_TransformerBlock_Encoder(nn.Module):
    def __init__(self, seq_length, block_size, embed_size, heads, dropout, forward_expansion, actf=nn.GELU, embed_block_size=None):
        super().__init__()
        assert 2**int(np.log2(block_size)) == block_size, 'Block size must be power of 2'
        assert 2**int(np.log2(seq_length)) == seq_length, 'Sequence length must be power of 2'
        
        if embed_block_size is not None:
            assert 2**int(np.log2(embed_block_size)) == embed_block_size, 'Embeddings block size must be power of 2'
        assert seq_length%block_size == 0, 'Sequence length must be divisible exactly by block_size'
        
        self.block_size = block_size
        self.seq_len = seq_length
        self.embed_block_size = embed_block_size
        
        def log_base(a, base):
            return np.log(a) / np.log(base)
        
        num_layers = int(np.ceil(log_base(seq_length, base=block_size)))
        self.sparse_transformers = []
        self.gaps = []
        for i in range(num_layers):            
            tr = TransformerBlock(embed_size, heads, dropout, forward_expansion, actf, embed_block_size)
            self.sparse_transformers.append(tr)
            ### find which permutation gives valid shape
            gap = self.block_size**i
            if gap*self.block_size <= self.seq_len:
                self.gaps.append(gap)
            else:
                self.gaps.append(int(np.ceil(self.seq_len/self.block_size)))
#                 break
            
        self.sparse_transformers = nn.ModuleList(self.sparse_transformers)
        

    def forward(self, x):
        N, seq_len, d_model = x.shape
        ### (N, seq_len, d_model) of the input x
        
        assert seq_len == self.seq_len, 'The sequence length of given input does not match this model'
            
        for i, fn in enumerate(self.sparse_transformers):
            gap = self.gaps[i]
            print(x.shape, 'before sparse')
            x = x.view(N, -1, self.block_size, gap, d_model).transpose(2, 3).contiguous().view(-1, self.block_size, d_model)
            ### This does not work with pytorch MSA layer because, the masking is done for full seq, cant do that on batchwise block
            x = fn(x)
            x = x.view(N, -1, gap, self.block_size, d_model).transpose(2, 3).contiguous()

        x = x.view(N, seq_len, -1)
        return x
    
    
#############################################################################
#############################################################################





#############################################################################
#############################################################################


    
    
#############################################################################
#############################################################################





#############################################################################
#############################################################################


    
    
#############################################################################
#############################################################################





#############################################################################
#############################################################################


    
    
#############################################################################
#############################################################################





#############################################################################
#############################################################################





