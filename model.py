import torch
import torch.nn as nn
from torch.nn import functional as F

def positional_encoding(seq_len, d_model, device='cpu'):
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (T, 1)
    i = torch.arange(d_model // 2, dtype=torch.float32, device=device).unsqueeze(0)  # (1, d/2)
    denom = torch.pow(10000, (2 * i) / d_model)  # (1, d/2)
    angles = pos / denom                         # (T, d/2)

    pe = torch.zeros((seq_len, d_model), device=device)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)

    return pe

class GPT(nn.Module):
  def __init__(self, vocab_size, block_size, embed_dim, num_heads, FFN_depth, encoder_layers):
    super().__init__()
    self.block_size = block_size
    self.FFN_depth = FFN_depth
    self.num_heads = num_heads
    self.embed_dim = embed_dim
    self.encoder_layers = encoder_layers
    self.token_emb = nn.Embedding(vocab_size, embed_dim)       # (v, d)
    self.register_buffer("pos_emb", positional_encoding(block_size, embed_dim))
    self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
    self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
    self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
    self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)
    self.Wkd = nn.Linear(embed_dim, embed_dim, bias=False)
    self.Wvd = nn.Linear(embed_dim, embed_dim, bias=False)
    self.lastW = nn.Linear(embed_dim, vocab_size, bias=False)
    self.LN1 = nn.LayerNorm(embed_dim)
    self.LN2 = nn.LayerNorm(embed_dim)
    self.ffn = nn.Sequential(
        nn.Linear(embed_dim, self.FFN_depth),
        nn.GELU(),
        nn.Linear(self.FFN_depth, embed_dim)
    )
  
  def layer_1(self, X):
    X = self.token_emb(X) * self.embed_dim ** 0.5      # (B, n) --> (B, n, d)
    T = X.shape[1]
    X = X + self.pos_emb[:T,:]
    return X                                                    # (B, n, d)    

  def single_head(self, X):
    q = self.Wq(X)
    k = self.Wk(X)
    v = self.Wv(X)
    return q, k, v
    
  def multi_head(self, q, k, v, mode):
    B, T, D = q.shape
    H = self.num_heads
    H_dim = int(D / self.num_heads)
    Q = q.view(B, T, H, H_dim).transpose(1, 2).contiguous()
    K = k.view(B, T, H, H_dim).transpose(1, 2).contiguous()
    V = v.view(B, T, H, H_dim).transpose(1, 2).contiguous()

    if mode == "decode":
      attention_scores = Q @ K.transpose(-2, -1) / H_dim ** 0.5
      mask = torch.tril(torch.ones(T, T, device=Q.device))
      attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
    else:
      attention_scores = Q @ K.transpose(-2, -1) / H_dim ** 0.5
    attention_probs = F.softmax(attention_scores, dim=-1)
    out = attention_probs @ V
    out = out.transpose(1, 2).contiguous().view(B, T, D)
    return self.Wo(out)

  def add_and_layernorm(self, layer_1, output, LN):
    residue = layer_1 + output
    return LN(residue)

  def encoder_layer(self, X):
    q, k, v = self.single_head(X)
    output = self.multi_head(q, k, v, "encode")
    layer2 = self.add_and_layernorm(X, output, self.LN1)
    ffn = self.ffn(layer2)
    return self.add_and_layernorm(layer2, ffn, self.LN2)

  def stack_encoder(self, X):
    Hl = X
    for _ in range(self.encoder_layers):
      Hl = self.encoder_layer(Hl)
    return Hl

  def decoder_layer(self, y, Hl1):
    # 1. Masked self-attention (decoder-to-decoder)
    q1, k1, v1 = self.single_head(y)
    self_attn_out = self.multi_head(q1, k1, v1, mode="decode")
    x = self.add_and_layernorm(y, self_attn_out, self.LN1)

    # 2. Cross-attention (decoder-to-encoder)
    q2 = self.Wq(x)
    k2 = self.Wkd(Hl1)
    v2 = self.Wvd(Hl1)
    cross_out = self.multi_head(q2, k2, v2, mode="encode")
    x = self.add_and_layernorm(x, cross_out, self.LN2)

    ffn_out = self.ffn(x)
    x = self.add_and_layernorm(x, ffn_out, self.LN2)
    return x

  def stack_decoder_layer(self, y, Hl1):
    y = self.token_emb(y) * self.embed_dim ** 0.5
    T = y.shape[1]
    y = y + self.pos_emb[:T, :]
    for _ in range(self.encoder_layers):
        y = self.decoder_layer(y, Hl1)
    return y

  def forward(self, x, y):
    x = self.token_emb(x) * self.embed_dim ** 0.5
    T = x.shape[1]
    x = x + self.pos_emb[:T, :]
    Hl1 = self.stack_encoder(x)
    output = self.stack_decoder_layer(y, Hl1)
    logits = self.lastW(output)
    return logits