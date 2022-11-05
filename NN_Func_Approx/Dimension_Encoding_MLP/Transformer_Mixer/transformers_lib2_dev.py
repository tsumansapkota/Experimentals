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


class SelfAttention_Sparse(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask, block_size):
        # Get number of training examples
        N = query.shape[0]
        
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len//block_size, block_size, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len//block_size, block_size, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len//block_size, block_size, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("n aq h d , n ak h d -> n h a qk", [queries, keys])
        # queries shape: (N, n_query_blocks, block_query_len, heads, heads_dim),
        # keys shape: (N, n_key_blocks, block_key_len, heads, heads_dim)
        # energy: (N, heads, n_query_blocks, block_query_len, block_key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=-1)
        # attention shape: (N, heads, num_blocks, query_len, key_len)

        out = torch.einsum("n h a q k , n a k hd -> n a q hd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, num_blocks, query_len, key_len)
        # values shape: (N, num_blocks, block_value_len, heads, heads_dim)
        # out after matrix multiply: (N, num_blocks, block_query_len, heads, head_dim), then
        # we reshape and flatten the (1,2)dimensions as well as (3,4) dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)
        return out
    

    
    
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
    def __init__(self, input_dim, layer_dims, actf=nn.GELU):
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
        # x = x.transpose(1,0).reshape(bs, -1)
        x = x.transpose(1,0).contiguous().view(bs, -1)
        
        return x
    
############################################################################
############################################################################

class BlockMLP_MixerBlock(nn.Module):
    
    def __init__(self, input_dim, block_dim, hidden_layers_ratio=[2], actf=nn.GELU):
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




class Sparse_TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, actf=nn.GELU, embed_block_size:int=None):
        super().__init__()
        
        self.attention = SelfAttention_Sparse(embed_size, heads)
            
        self.norm1 = nn.LayerNorm(embed_size)

        
        if embed_block_size is None or embed_block_size == embed_size:
            hid = int(forward_expansion * embed_size)
            self.feed_forward = ResMlpBlock(embed_size, [forward_expansion], actf)
        else:
            self.feed_forward = BlockMLP_MixerBlock(embed_size, embed_block_size, [forward_expansion], actf)
            
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, block_size):
        attention = self.attention(value, key, query, mask, block_size)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        
        _xs = x.shape
        x = x.view(-1, _xs[-1])
        
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward))
        return out.view(*_xs)


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
            tr = Sparse_TransformerBlock(embed_size, heads, dropout, forward_expansion, actf, embed_block_size)
            self.sparse_transformers.append(tr)
            ### find which permutation gives valid shape
            gap = self.block_size**i
            if gap*self.block_size <= self.seq_len:
                self.gaps.append(gap)
            else:
                self.gaps.append(int(np.ceil(self.seq_len/self.block_size)))
#                 break
            
        self.sparse_transformers = nn.ModuleList(self.sparse_transformers)
        

    def forward(self, x, mask = None):
        N, seq_len, d_model = x.shape
        ### (N, seq_len, d_model) of the input x
        
        assert seq_len == self.seq_len, 'The sequence length of given input does not match this model'
            
        for i, fn in enumerate(self.sparse_transformers):
            gap = self.gaps[i]
#             print(i, gap)
            x = x.view(N, -1, self.block_size, gap, d_model).transpose(2, 3).contiguous().view(N, seq_len, d_model)
            x = fn(x, x, x, mask, self.block_size)
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





