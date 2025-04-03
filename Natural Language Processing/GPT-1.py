import torch
import torch.nn as nn
from einops import rearrange

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.scale = torch.sqrt(torch.tensor(d_model / n_heads))

    def forward(self, Q, K, V, mask = None):
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        Q = rearrange(Q, 'n w (h d) -> n h w d', h = self.n_heads)
        K = rearrange(K, 'n w (h d) -> n h w d', h = self.n_heads)
        V = rearrange(V, 'n w (h d) -> n h w d', h = self.n_heads)

        attention_score = Q @ K.transpose(-2, -1)/self.scale

        if mask is not None:
            attention_score[mask] = -1e9
        
        attention_weights = torch.softmax(attention_score, dim = -1)

        attention = attention_weights @ V

        x = rearrange(attention, 'n h w d -> n w (h d)')
        x = self.fc_o(x)

        return x, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, drop_p):
        super().__init__()

        self.linear = nn.Sequential(nn.Linear(d_model, d_ff),
                                    nn.GELU(), #different activation function compared to transformer
                                    nn.Dropout(drop_p),
                                    nn.Linear(d_ff, d_model))
    
    def forward(self, x):
        x = self.linear(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, drop_p):
        super().__init__()

        self.self_atten = MHA(d_model, n_heads)
        self.self_atten_LN = nn.LayerNorm(d_model)

        self.FF = FeedForward(d_model, d_ff, drop_p)
        self.FF_LN = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_p)
        
    def forward(self, x, mask):
        residual, atten_enc = self.self_atten(x, x, x, mask)
        residual = self.dropout(residual)
        x = self.self_atten_LN(residual + x)

        residual = self.FF(x)
        residual = self.dropout(residual)
        x = self.FF_LN(residual + x)

        return x, atten_enc

class Decoder(nn.Module):
    def __init__(self, input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p):
        super().__init__()

        self.scale = torch.sqrt(torch.tensor(d_model))
        self.input_embedding = input_embedding
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.dropout = nn.Dropout(drop_p)

        self.layers = nn.ModuleList([DecoderLayer(n_heads, d_model, d_ff, drop_p) for _ in range(n_layers)])
    
    def forward(self, src, mask, atten_map_save = False):
        pos = torch.arange(src.shape[1]).expand_as(src).to(DEVICE)

        x = self.scale * self.input_embedding(src) + self.pos_embedding(pos)
        x = self.dropout(x)

        atten_encs = torch.tensor([]).to(DEVICE)
        for layer in self.layers:
            x, atten_enc = layer(x, mask)
            if atten_map_save is True:
                atten_encs = torch.cat([atten_encs, atten_enc[0].unsqueeze(0)], dim = 0)
        
        return x, atten_encs
    
class GPT(nn.Module):
    def __init__(self, 
                 vocab_size = 30522, 
                 max_len = 512, 
                 n_layers = 12, 
                 d_model = 768, 
                 d_ff = 3072, 
                 n_heads = 12, 
                 drop_p = 0.1, 
                 pad_idx = 0):
        super().__init__()

        self.input_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder = Decoder(self.input_embedding, max_len, n_layers, d_model, d_ff, n_heads, drop_p)

        self.n_heads = n_heads
        self.pad_idx = pad_idx

    def make_mask(self, src):
        pad_mask = (src == self.pad_idx).unsqueeze(1).unsqueeze(2)
        pad_mask = pad_mask.expand(src.shape[0], self.n_heads, src.shape[1], src.shape[1])
        
        seq_len = src.shape[1]
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), device = DEVICE), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) 
        
        mask = pad_mask | causal_mask
        
        return mask
    
    def forward(self, src):
        mask = self.make_mask(src)
        out, atten_encs = self.decoder(src, mask)

        return out, atten_encs
