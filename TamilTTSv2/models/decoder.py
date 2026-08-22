import torch
import torch.nn as nn

try:
    from .modules import PositionalEncoding, FFTBlock
except ImportError:
    from modules import PositionalEncoding, FFTBlock


class MelDecoder(nn.Module):
    """
    Pre-LN Transformer mel decoder with per-block FiLM style conditioning.
    """
    def __init__(self, hidden_dim=512, num_layers=4, num_heads=8, ff_dim=1024,
                 style_dim=256, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        self.blocks = nn.ModuleList([
            FFTBlock(hidden_dim, num_heads, ff_dim, style_dim=style_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_expanded, style, mel_mask_true_is_pad=None):
        """
        x_expanded: [B, Tm, H] duration-expanded hidden states
        style:      [B, S]
        Returns [B, Tm, H]
        """
        h = self.pos_encoder(x_expanded)
        for block in self.blocks:
            h = block(h, style=style, key_padding_mask=mel_mask_true_is_pad)
        h = self.norm(h)

        if mel_mask_true_is_pad is not None:
            h = h.masked_fill(mel_mask_true_is_pad.unsqueeze(-1), 0.0)
        return h
