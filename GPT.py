import torch
import torch.nn as nn
from torch.nn import functional as F


'''
Self-Attention Head
'''
class SelfAttentionHead(nn.Module):
    def __init__(self, embd_dim: int, context_size: int, query_dim: int, dropout: float):
        super().__init__()
        self.query_dim = query_dim
        self.Q = nn.Linear(embd_dim, self.query_dim, bias = False) # QUERY MATRIX, (T, F), F = query dimension/feature size
        self.K = nn.Linear(embd_dim, self.query_dim, bias = False) # KEY MATRIX, (T, F)
        self.V = nn.Linear(embd_dim, self.query_dim, bias = False) # VALUE MATRIX, (T, F)
        # Key vectors corresponding to later tokens in context cannot attend to query vectors corresponding to earlier tokens.
        # This is because GPTs are predictive modelsL they learn contextual relationships between tokens in order to better
        # predict/generate later tokens given earlier ones. This buffer hides later keys from earlier queries.
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x: torch.Tensor): # x is (B, T, C) = (batch size, token-sequence length, embedding dimension)
        self.tril = self.tril.to(x.device)
        T = x.shape[1]

        queries = self.Q(x) # (B, T, F)
        keys = self.K(x) # (B, T, F)
        weights = queries @ keys.transpose(-2, -1) * self.query_dim ** -0.5 # (B, T, T), normalized
        # Slice self.tril by T to handle sequence lengths smaller than context size during generation.
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # replace 0 weights with -infinity
        weights = F.softmax(weights, dim=-1) # each column of weights -> prob distribution, one for each querying token
        # Dropout occurs after weights are calculated because this portion learns/utilized patterns and relationships.
        # Dropout does not occur after the final output is calculated since the (theoretical) purpose of the value
        # matrix and of multiplying the weight and value matrices is to combine learned patterns and to put the data
        # in a usable form for the next stage.
        weights = self.dropout(weights)
        
        values = self.V(x) # (B, T, F)
        out = weights @ values # B many (T, T) @ (T, F) = (B, T, F)
        return out # return changes to embeddings


'''
Multi-Headed Self-Attention Layer
'''
class MultiHeadAttention(nn.Module):
    def __init__(self, embd_dim: int, context_size: int, num_heads: int, query_dim: int, dropout: float):
        super().__init__()
        attention_heads = [SelfAttentionHead(embd_dim, context_size, query_dim, dropout) for _ in range(num_heads)]
        # nn.ModuleList does not enforce sequential forward passes among its modules, unlike nn.Sequential.
        # So, use nn.ModuleList because self-attention heads process separate contents in parallel.
        self.attention_heads = nn.ModuleList(attention_heads)
        # Combine outputs from each attention head into single coherant representation.
        # If the number of attention heads times the query dimension equals the embedding dimension then representation
        # layer is not strictly necessary but still improves performance. 
        self.representation_layer = nn.Linear(num_heads * query_dim, embd_dim) # also called output matrix
        
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, x: torch.Tensor): # x is (B, T, C)
        out = torch.cat([h(x) for h in self.attention_heads], dim=-1)
        out = self.representation_layer(out)
        out = self.dropout(out)
        return out


'''
Feedforward Layer
'''
class FeedForward(nn.Module):
    def __init__(self, embd_dim: int, proj_factor: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embd_dim, proj_factor * embd_dim), # project up to then down from a higher dimension to learn details
            nn.ReLU(), # doesn't have to be ReLU
            nn.Linear(embd_dim * proj_factor, embd_dim),
            nn.Dropout(dropout))
        
        
    def forward(self, x: torch.Tensor):
        return self.net(x)
        

'''
Decoder Block
'''
class Decoder(nn.Module): # decoder block class
    def __init__(self, embd_dim: int, context_size: int, num_heads: int, fdfwd_proj_factor: int, dropout: float):
        super().__init__()
        query_dim = embd_dim // num_heads # number of semantic features captured by single attention head
        self.attention = MultiHeadAttention(embd_dim, context_size, num_heads, query_dim, dropout)
        self.feedforward = FeedForward(embd_dim, fdfwd_proj_factor, dropout)
        self.ln1 = nn.LayerNorm(embd_dim) # nn.LayerNorm automatically normalizes along the last dimension; it just needs the size
        self.ln2 = nn.LayerNorm(embd_dim)
        
        
    def forward(self, x: torch.Tensor):
        y = self.attention.forward(x)
        x = self.ln1(x + y) # modify embeddings via multi-head self-attention then normalize
        y = self.feedforward.forward(x)
        x = self.ln2(x + y) # modify embeddings via feedforward then normalize
        return x


'''
GPT Model
'''
class GPT(nn.Module):
    def __init__(self, vocab_size: int, embd_dim: int, context_size: int, num_decoders: int, num_heads: int, dropout=0.2):
        super().__init__()
        self.context_size = context_size # for simplicity, context size is constant between training and evaluation

        self.token_embedding_table = nn.Embedding(vocab_size, embd_dim)
        self.position_embedding_table = nn.Embedding(self.context_size, embd_dim)

        decoders = [Decoder(embd_dim, self.context_size, num_heads, 4, dropout) for _ in range(num_decoders)]
        self.decoder_blocks = nn.Sequential(*decoders)

        self.layer_norm_final = nn.LayerNorm(embd_dim)
        self.linear = nn.Linear(embd_dim, vocab_size)
        
        self.apply(self.init_weights) # applies given function to all nn modules within caller
        
        
    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        
    def forward(self, indices: torch.Tensor, targets=None):
        token_embedding = self.token_embedding_table(indices) # gives B by T many embeddings of size C
        position_embedding = self.position_embedding_table(torch.arange(indices.shape[1], device=indices.device)) # (T, C)

        x = token_embedding + position_embedding # combine embedding information, position_embedding is broadcast during addition
        x = self.decoder_blocks(x)
        x = self.layer_norm_final(x)
        logits = self.linear(x)
        
        if targets is None: # not training
            loss = None
        else: # training
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # reshape to list of (transformed) embedding vectors
            targets = targets.view(B*T) # single target for each of the B*T tokens
            loss = F.cross_entropy(logits, targets) # loss is a value AND a tensor storing the model's current parameter info
            
        return logits, loss
    

    def generate(self, indices: torch.Tensor, num_new_tokens: int): # indices is (B, T) tensor of vocab indices
        for _ in range(num_new_tokens):
            # Crop token sequences of input to context-size most recent tokens.
            # This is done so that, in long text generation, when indices is modified and fed back this method, if indices
            # becomes longer than allowed context length, only the most recent tokens within that length are considered.
            indices_recent = indices[:, -self.context_size:]
            logits = self.forward(indices_recent)[0] # just logits, not loss
            logits = logits[:, -1, :] # logits for next token at last timestep of each sequence (B, 1, C)
            probs = F.softmax(logits, dim=-1) # logits --> probabilities (B, 1, 1)
            index_next = torch.multinomial(probs, num_samples=1) # Sample from distribution for next token(s) (B, 1)
            indices = torch.cat((indices, index_next), dim=1) # (B, T+1)
        return indices