import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model:int, vocab_size: int,):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # create matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len)
        # PE(pos, 2i) = sin(pos/(10000^(2i/d_model)))
        # PE(pos, 2i+1) = cos(pos/(10000^(2i/d_model)))
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # apply the sin to even positions
        # start from 0, upto end, forward by 2, i.e, 0, 2, 4...
        pe[:, 0::2] = torch.sin(position*div_term) 
        # start from 1, upto end, forward by 2, i.e, 1, 3, 5...
        pe[:, 1::3] = torch.cos(position*div_term)

        # add batch dimension to the pe to apply for batch of sentences, instead of one sentence
        pe = pe.unsqueeze(0) # becomes (1, seq_len, d_model)

        ''' register the pe as buffer, to keep it in model, not as a learned parameter, 
        but as a vector when we save module. Registering it as a buffer is important
        '''
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x+ (self.pe[:, :x.shape[1], :]).requires_grad_(False) # tells this particular is not learned
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdiM=True)
        return self.alpha * (x-mean)/math.sqrt(std + self.eps) +self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, Seq_len, d_model) ==> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.linear_1(x)))
