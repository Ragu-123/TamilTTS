import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FramePositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for Mel-frame sequences."""
    def __init__(self, d_model, max_len=4000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """x: [B, T_mel, d_model]"""
        return x + self.pe[:, :x.size(1), :]


class ProsodyBlock(nn.Module):
    """
    Style-conditioned 1D Residual Convolution block with LeakyReLU.
    Uses dilated convolutions for wide temporal receptive field across neighboring phonemes.
    """
    def __init__(self, channels, kernel_size=5, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        res = x
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.1)
        x = self.norm2(self.conv2(x))
        x = F.leaky_relu(x + res, 0.1)
        return x


class DiffusionProsody(nn.Module):
    """
    Prosody & Acoustic Mel Representation Network.
    Modulates text representation with Frame-level Positional Encodings and Speaker Style.
    """
    def __init__(self, in_channels=512, style_dim=256, hidden_channels=512, num_blocks=8):
        super().__init__()
        self.frame_pe = FramePositionalEncoding(in_channels)
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels, hidden_channels * 2),
        )
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 5, padding=2)
        
        # Exponentially dilated convolution blocks for full sentence-level receptive field
        dilations = [1, 2, 4, 8, 1, 2, 4, 8]
        self.blocks = nn.ModuleList([
            ProsodyBlock(hidden_channels, kernel_size=5, dilation=dilations[i % len(dilations)])
            for i in range(num_blocks)
        ])
        self.conv_out = nn.Conv1d(hidden_channels, in_channels, 5, padding=2)
        self.norm_out = nn.LayerNorm(in_channels)

    def forward(self, x, style):
        """
        x:     [B, T_mel, hidden_dim]
        style: [B, style_dim]
        """
        # 1. Add Frame-level Positional Encoding (enables dynamic pitch & formant transitions)
        x_pos = self.frame_pe(x)

        # 2. Speaker style scale and shift
        style_params = self.style_proj(style).unsqueeze(-1)  # [B, 2*H, 1]
        scale, shift = style_params.chunk(2, dim=1)         # [B, H, 1] each

        h = self.conv_in(x_pos.transpose(1, 2))              # [B, H, T_mel]
        h = h * (1.0 + torch.tanh(scale)) + shift            # FiLM modulation

        for block in self.blocks:
            h = block(h)

        h = self.conv_out(h).transpose(1, 2)                 # [B, T_mel, H]
        out = self.norm_out(x_pos + h)                       # Residual connection
        return out
