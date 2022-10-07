import torch
import torch.nn as nn
import numpy as np


#############################################################################
#############################################################################

class MlpBLock(nn.Module):
    
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
        return self.mlp(x)
    
    
#############################################################################
#############################################################################
    
    
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
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

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out
    
    
#############################################################################
#############################################################################


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, actf=nn.GELU):
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, int(forward_expansion * embed_size)),
            actf(),
            nn.Linear(int(forward_expansion * embed_size), embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query):
        attention = self.attention(query, query, query, None)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


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

class ViT_Classifier(nn.Module):
    
    def __init__(self, image_dim:tuple, patch_size:tuple, hidden_expansion:float, num_blocks:int, num_classes:int, pos_emb = True):
        super().__init__()
        
        self.img_dim = image_dim ### must contain (C, H, W) or (H, W)
        
        ### find patch dim
        d0 = int(image_dim[-2]/patch_size[0])
        d1 = int(image_dim[-1]/patch_size[1])
        assert d0*patch_size[0]==image_dim[-2], "Image must be divisible into patch size"
        assert d1*patch_size[1]==image_dim[-1], "Image must be divisible into patch size"
#         self.d0, self.d1 = d0, d1 ### number of patches in each axis
        __patch_size = patch_size[0]*patch_size[1]*image_dim[0] ## number of channels in each patch
    
        ### find channel dim
        channel_size = d0*d1 ## number of patches
        
        ### after the number of channels are changed
        init_dim = __patch_size
        final_dim = int(__patch_size*hidden_expansion/2)*2
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        #### rescale the patches (patch wise image non preserving transform, unlike bilinear interpolation)
        self.channel_change = nn.Linear(init_dim, final_dim)
        print(f"ViT Mixer : Channes per patch -> Initial:{init_dim} Final:{final_dim}")
        
        
        self.channel_dim = final_dim
        self.patch_dim = channel_size
        
        self.transformer_blocks = []
        
        f = self.get_factors(self.channel_dim)
        print(f)
        ### get number of heads close to the square root of the channel dim
        ### n_model = n_heads*heads_dim (= channel dim)
        fi = np.abs(np.array(f) - np.sqrt(self.channel_dim)).argmin()
        
        _n_heads = f[fi]
        
        print(self.channel_dim, _n_heads)
        for i in range(num_blocks):
            L = TransformerBlock(self.channel_dim, _n_heads, 0, 2)
            self.transformer_blocks.append(L)
        self.transformer_blocks = nn.Sequential(*self.transformer_blocks)
        
        self.linear = nn.Linear(self.patch_dim*self.channel_dim, num_classes)
        
        self.positional_encoding = nn.Identity()
        if pos_emb:
            self.positional_encoding = PositionalEncoding(self.channel_dim, dropout=0)
    
    def get_factors(self, n):
        facts = []
        for i in range(2, n+1):
            if n%i == 0:
                facts.append(i)
        return facts    
        
    def forward(self, x):
        bs = x.shape[0]
        x = self.unfold(x).swapaxes(-1, -2)
        x = self.channel_change(x)
        x = self.positional_encoding(x)
        x = self.transformer_blocks(x)
        x = self.linear(x.view(bs, -1))
        return x




#############################################################################
#############################################################################


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
    
    
#############################################################################
#############################################################################


class Sparse_TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, actf=nn.GELU):
        super().__init__()
        
        self.attention = SelfAttention_Sparse(embed_size, heads)
            
        self.norm1 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            actf(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask, block_size):
        attention = self.attention(value, key, query, mask, block_size)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


#############################################################################
#############################################################################


class Mixer_TransformerBlock_Encoder(nn.Module):
    def __init__(self, seq_length, block_size, embed_size, heads, dropout, forward_expansion, actf=nn.GELU):
        super().__init__()
        assert 2**int(np.log2(block_size)) == block_size, 'Block size must be power of 2'
        assert 2**int(np.log2(seq_length)) == seq_length, 'Sequence length must be power of 2'
        assert seq_length%block_size == 0, 'Sequence length must be divisible exactly by block_size'
        
        self.block_size = block_size
        self.seq_len = seq_length
        
        def log_base(a, base):
            return np.log(a) / np.log(base)
        
        num_layers = int(np.ceil(log_base(seq_length, base=block_size)))
        self.sparse_transformers = []
        self.gaps = []
        for i in range(num_layers):            
            tr = Sparse_TransformerBlock(embed_size, heads, dropout, forward_expansion, actf)
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





