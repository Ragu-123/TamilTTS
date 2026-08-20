import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    10-layer Transformer Text Encoder for Tamil characters/tokens.
    Includes learned token embeddings, LayerNorm, and src_key_padding_mask support.
    """
    def __init__(self, vocab_size=256, hidden_dim=512, num_layers=10, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
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
        h = self.norm(self.embedding(x))
        if mask is not None:
            h = self.encoder(h, src_key_padding_mask=mask)
            h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        else:
            h = self.encoder(h)
        return h
