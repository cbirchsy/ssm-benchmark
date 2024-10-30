import torch
from torch import nn
from mamba_ssm import Mamba as MambaLayer


from models import MATCH

class GLU(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim * 2)
    def forward(self, x):
        out = self.linear(x)
        return out[:, :, :x.shape[2]] * torch.sigmoid(out[:, :, x.shape[2]:])

class MambaBlock(torch.nn.Module):
    def __init__(self, hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm, bidirectional=True, bidirectional_strategy='concat', tie_weights=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.tie_weights = tie_weights
        if self.bidirectional_strategy == 'concat':
            self.dim_reduce = nn.Linear(2*hidden_dim, hidden_dim)
        self.mamba = MambaLayer(d_model=hidden_dim, d_state=state_dim, d_conv=conv_dim, expand=expansion)
        if not self.tie_weights:
            self.mamba_rev = MambaLayer(d_model=hidden_dim, d_state=state_dim, d_conv=conv_dim, expand=expansion)
        if glu:
            self.glu = GLU(hidden_dim)
        else:
            self.glu = None
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if norm in ["layer"]:
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm in ["batch"]:
            # TODO: add batch norm
            raise RuntimeError("dimensions don't agree for batch norm to work")
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm==None:
            self.norm = nn.Identity()
        self.prenorm = prenorm
        
    def forward(self, x):
        skip = x
        if self.prenorm:
            x = self.norm(x)
        if self.bidirectional:
            x_f = self.mamba(x)
            if not self.tie_weights:
                x_r = self.mamba_rev(x.flip(dims=(1,))).flip(dims=(1,))
            else:
                x_r = self.mamba(x.flip(dims=(1,))).flip(dims=(1,))
            if self.bidirectional_strategy == 'concat':
                x = torch.cat((x_f,x_r), axis=-1) # concat in embedding dim like S5
                x = self.dim_reduce(x) # linear layer to reduce dims
            elif self.bidirectional_strategy == 'add':
                x = x_f + x_r # add embeddings (reverse invariant)
            elif self.bidirectional_strategy == 'ew_multiply':
                x = x_f * x_r # element-wise multiply embeddings (reverse invariant)
            else:
                raise NotImplementedError(f"bidirectional_strategy='{self.bidirectional_strategy}' not implemented.")
        else:
            x = self.mamba(x)
        x = self.dropout(self.activation(x))
        if self.glu is not None:
            x = self.glu(x)
        x = self.dropout(x)
        x = x + skip
        if not self.prenorm:
            x = self.norm(x)
        return x
    
class Mamba(torch.nn.Module):
    def __init__(self, num_blocks, input_dim, output_dim, hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm, dual, pooling="mean", 
                 bidirectional=True, bidirectional_strategy='concat', tie_weights=True, tokenized_inputs=False):
        super().__init__()
        self.tokenized_inputs = tokenized_inputs
        if self.tokenized_inputs:
            self.encoder = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[MambaBlock(hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm, 
                                                 bidirectional=bidirectional, bidirectional_strategy=bidirectional_strategy, tie_weights=tie_weights
                                                ) for _ in range(num_blocks)])
        self.linear_decoder = nn.Linear(hidden_dim, output_dim)
        self.pooling = pooling
        self.softmax = nn.LogSoftmax(dim=1)
        self.dual = dual
        if dual:
            self.match = MATCH(output_dim*2, output_dim)
            
    def forward(self, x):
        x = self.encoder(x)
        x = self.blocks(x)
        if self.pooling in ["mean"]:
            x = torch.mean(x, dim=1)
        elif self.pooling in ["max"]:
            x = torch.max(x, dim=1)[0]
        elif self.pooling in ["last"]:
            x = x[:,-1,:]
        else:
            x = x # no pooling
        x = self.linear_decoder(x)
        if self.dual:
            (x1, x2) = torch.split(x, int(x.shape[0]/2))
            x = self.match(torch.concatenate((x1, x2), dim=1))
        return torch.softmax(x, dim=1)
