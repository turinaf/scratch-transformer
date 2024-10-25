import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size: int,):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float) :
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len)
        # PE(pos, 2i) = sin(pos/(10000^(2i/d_model)))
        # PE(pos, 2i+1) = cos(pos/(10000^(2i/d_model)))
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # apply the sin to even positions
        # start from 0, upto end, forward by 2, i.e, 0, 2, 4...
        pe[:, 0::2] = torch.sin(position*div_term) # sin(position * (10000 ** (2i / d_model))
        # start from 1, upto end, forward by 2, i.e, 1, 3, 5...
        pe[:, 1::2] = torch.cos(position*div_term)  # cos(position * (10000 ** (2i / d_model))

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
    def __init__(self, features: int, eps:float = 10**-6) -> None:
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(features)) # alpha is learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter
    
    def forward(self, x):
        if x is None:
            raise ValueError("Input to LayerNormalization is None")
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")
        if x.dim() == 0:
            raise ValueError("Input tensor must have at least one dimension")
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x-mean)/torch.sqrt(std + self.eps) +self.bias
    

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) :
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (Batch, Seq_len, d_model) ==> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(self.linear_1(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # d_k is the last dimension of query, key and value
        d_k = query.shape[-1]
        # (Batch, h, seq_len, d_k) --> (Batch, he, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1))/ math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        '''If we want to prevent some words to intract with eachother, we mask them'''
        query = self.w_q(q) # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        key = self.w_k(k)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        # (batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        # h * d_k = d_model
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # do in place
        # return x multiplied by w_o, which is output matrix
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    def __init__(self, features:int, dropout:float) :
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        # sublayer is previous layer
        if x is None:
            raise ValueError("Input to Residual connection is None")
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connectins = nn.ModuleList(ResidualConnection(features, dropout) for _ in range(2))

    def forward(self, x, src_mask):
        if x is None: 
            raise ValueError("Input to EncoderBlock is None")
        x = self.residual_connectins[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connectins[1](x, self.feed_forward_block)
        return x

# Nx EncoderBlock    
class Encoder(nn.Module):
    def __init__(self, features: int, layers:nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        if x is None: 
            raise ValueError("Input to Encoder block is None")
        for i, layer in enumerate(self.layers):
            # print(f"EncoderBlock {i}/{len(self.layers)}: \n X = {x}\n")
            if x is None: 
                raise ValueError(f"X is non in EncoderBlock {i}, which is output of EncoderBlock {i-1}")
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderBlock(nn.Module):
    def __init__(self, features:int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        if x is None:
            raise ValueError("Input to DecoderBlock is None!")
        # self attention part of the decoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # self attention part, where query and value comes from encoder
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

# Nx Decoder Block
class Decoder(nn.Module):
    def __init__(self,features:int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        if x is None: 
            raise ValueError("Input to Decoder Block is None")
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim = -1)
    

# Transformer block
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed:InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) :
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        assert src is not None, "Input to encoder is None"
        src = self.src_embed(src) # apply embedding to src sentences
        assert src is not None, "src is is None after embedding"
        src = self.src_pos(src) # the positional encoding
        assert src is not None, "src is is None after positional encoding"
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        assert tgt is not None, "Input to decoder is None"
        tgt = self.tgt_embed(tgt) # apply embedding to target sentences
        assert tgt is not None, "tgt is is None after embedding"
        tgt = self.tgt_pos(tgt)
        assert tgt is not None, "tgt is is None after positional encoding"
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
    

# build transformer and initialize parameters with some initial values
def build_transformer(src_vocab_size:int, tgt_vocab_size:int, src_seq_len:int, tgt_seq_len: int, d_model=512, N:int = 6, h:int = 8, dropout:float = 0.1, d_ff:int =2048) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create4 the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model=d_model, d_ff=d_ff, dropout=dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and the decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # creat projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # create transformer 
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer