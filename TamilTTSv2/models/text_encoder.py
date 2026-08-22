import torch
import torch.nn as nn

try:
    from .modules import PositionalEncoding
except ImportError:
    from modules import PositionalEncoding


class TextEncoder(nn.Module):
    """
    Transformer text encoder: Embedding(padding_idx=0) + PosEnc + LN
    + nn.TransformerEncoder with src_key_padding_mask; pad positions zeroed.
    """
    def __init__(self, vocab_size=384, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x_ids, mask_bool_true_is_pad=None):
        """
        x_ids:              [B, T_text] token IDs
        mask_bool_true_is_pad: [B, T_text] True = PAD (optional)
        Returns [B, T_text, hidden_dim]
        """
        h = self.embedding(x_ids)
        h = self.pos_encoder(h)
        h = self.norm(h)

        if mask_bool_true_is_pad is not None:
            h = self.encoder(h, src_key_padding_mask=mask_bool_true_is_pad)
            h = h.masked_fill(mask_bool_true_is_pad.unsqueeze(-1), 0.0)
        else:
            h = self.encoder(h)
        return h
