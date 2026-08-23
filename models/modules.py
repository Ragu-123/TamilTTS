import math

import torch
import torch.nn as nn


def sequence_mask(lens, max_len=None):
    """
    lens: [B] int/long tensor of valid lengths.
    Returns BoolTensor [B, max_len] where True means index < lens[i].
    """
    if max_len is None:
        max_len = int(lens.max().item())
    idx = torch.arange(max_len, device=lens.device).unsqueeze(0)  # [1, L]
    return idx < lens.unsqueeze(1)  # [B, L]


def length_regulate(x, durs, max_len=None):
    """
    Fully vectorized duration expansion.
    x:    [B, T, H] float
    durs: [B, T] float durations
    Returns [B, max_len, H], zero-padded / truncated to max_len.
    """
    B, T, H = x.shape
    device = x.device

    reps = torch.round(durs).clamp(min=0).long()       # [B, T]
    out_lens = reps.sum(dim=1)                         # [B]

    # Source time-step index for every expanded frame (no loops).
    ar = torch.arange(T, device=device).unsqueeze(0).expand(B, T)   # [B, T]
    src_idx = torch.repeat_interleave(ar.reshape(B * T), reps.reshape(B * T))
    batch_idx = torch.repeat_interleave(
        torch.arange(B, device=device), out_lens
    )                                                  # [total]

    total = src_idx.numel()
    offsets = torch.cumsum(out_lens, dim=0) - out_lens  # exclusive cumsum [B]
    pos_in_row = torch.arange(total, device=device) - offsets[batch_idx]

    if max_len is None:
        out_len = int(out_lens.max().item()) if total > 0 else 0
    else:
        out_len = max_len

    out = x.new_zeros(B, out_len, H)
    if total > 0 and out_len > 0:
        valid = pos_in_row < out_len
        out[batch_idx[valid], pos_in_row[valid]] = x.reshape(B * T, H)[src_idx[valid]]
    return out


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (added to input)."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: style -> (gamma, beta).
    Small-random weight init: modulation starts near-identity while still
    letting gradients flow into the style vector on the first backward
    (exact zero-init would cut the style gradient entirely).
    """
    def __init__(self, style_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(style_dim, 2 * hidden_dim)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, style):
        """
        x:     [B, T, H]
        style: [B, S]
        Returns x * (1 + gamma) + beta
        """
        gamma, beta = self.proj(style).unsqueeze(1).chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta


class ConvFF(nn.Module):
    """
    Pre-LN convolutional feedforward sublayer body (residual added outside).
    LN -> Conv1d(k3) GELU -> Conv1d(k3) -> Dropout
    """
    def __init__(self, hidden_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.w_1 = nn.Conv1d(hidden_dim, ff_dim, kernel_size=3, padding=1)
        self.w_2 = nn.Conv1d(ff_dim, hidden_dim, kernel_size=3, padding=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, T, H] -> [B, T, H]"""
        residual_normed = self.norm(x)
        y = residual_normed.transpose(1, 2)
        y = self.w_2(torch.nn.functional.gelu(self.w_1(y)))
        y = self.drop(y)
        return y.transpose(1, 2)


class FFTBlock(nn.Module):
    """
    Pre-LN Transformer block with ConvFF and FiLM style conditioning.
    FiLM is applied to the attention-sublayer output before its residual add,
    then ConvFF sublayer follows with its own residual.
    """
    def __init__(self, hidden_dim, num_heads, ff_dim, style_dim=None, dropout=0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = ConvFF(hidden_dim, ff_dim, dropout)
        self.film = FiLM(style_dim, hidden_dim) if style_dim is not None else None

    def forward(self, x, style=None, key_padding_mask=None):
        """
        x:                [B, T, H]
        style:            [B, S] or None
        key_padding_mask: [B, T] True=pad or None
        """
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        if self.film is not None and style is not None:
            attn_out = self.film(attn_out, style)
        x = x + attn_out
        x = x + self.ff(x)
        return x
