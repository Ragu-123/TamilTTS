import math

import torch
import torch.nn as nn

try:
    from .modules import PositionalEncoding
except ImportError:
    from modules import PositionalEncoding


class TextEncoder(nn.Module):
    """
    Transformer text encoder (pre-LN, standard wiring):
      Embedding(padding_idx=0) * sqrt(d) -> PosEnc(+dropout)
      -> nn.TransformerEncoder(norm_first=True, final LayerNorm, src_key_padding_mask)
    Pad positions zeroed after encoding.

    Previous version applied PosEnc *before* a standalone LayerNorm (attenuating the
    positional signal) on post-LN encoder layers with no final norm. This matches the
    decoder's pre-LN FFTBlock philosophy and the FastSpeech/Kokoro convention.
    """
    def __init__(self, vocab_size=384, hidden_dim=512, num_layers=6, num_heads=8,
                 ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim if ff_dim else hidden_dim * 4
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )

    def forward(self, x_ids, mask_bool_true_is_pad=None):
        """
        x_ids:              [B, T_text] token IDs
        mask_bool_true_is_pad: [B, T_text] True = PAD (optional)
        Returns [B, T_text, hidden_dim]
        """
        h = self.embedding(x_ids) * math.sqrt(self.hidden_dim)
        h = self.pos_encoder(h)

        if mask_bool_true_is_pad is not None:
            h = self.encoder(h, src_key_padding_mask=mask_bool_true_is_pad)
            h = h.masked_fill(mask_bool_true_is_pad.unsqueeze(-1), 0.0)
        else:
            h = self.encoder(h)
        return h
