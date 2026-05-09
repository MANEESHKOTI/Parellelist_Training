import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size=50257, max_seq_len=1024, d_model=768, n_layers=12, n_heads=12, d_ff=3072, dropout=0.1, tie_weights=True):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights: self.lm_head.weight = self.token_embedding.weight
        self.max_seq_len = vocab_size

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_embedding(idx) + self.pos_embedding(torch.arange(T, device=idx.device))
        for block in self.blocks: x = block(x)
        logits = self.lm_head(self.ln_f(x))
        
        loss = None
        if targets is not None:
            # FIX: Shift logits and labels so the model predicts the NEXT token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return logits, loss

class DecoderBlock(nn.Module):
    def __init__(self, d, nh, df, drp):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, dropout=drp, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, df), nn.GELU(), nn.Linear(df, d), nn.Dropout(drp))
        
    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False, is_causal=True)
        x = x + attn_out
        return x + self.ffn(self.ln2(x))
