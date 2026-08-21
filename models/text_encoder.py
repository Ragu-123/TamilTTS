import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer sequences."""
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: [B, T, d_model]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TextEncoder(nn.Module):
    """
    10-layer Transformer Text Encoder for Tamil characters/tokens.
    Includes learned token embeddings, Sinusoidal Positional Encoding,
    LayerNorm, and src_key_padding_mask support.
    """
    def __init__(self, vocab_size=256, hidden_dim=512, num_layers=10, num_heads=8, max_len=2000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1, max_len=max_len)
        self.norm = nn.LayerNorm(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        """
        x:    [B, T_text] token IDs
        mask: [B, T_text] boolean mask where True = PAD token
        """
        h = self.embedding(x)
        h = self.pos_encoder(h)
        h = self.norm(h)
        
        if mask is not None:
            h = self.encoder(h, src_key_padding_mask=mask)
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        else:
            h = self.encoder(h)
        return h
