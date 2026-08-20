import torch
import torch.nn as nn
import torch.nn.functional as F


class ProsodyBlock(nn.Module):
    """
    Style-conditioned 1D Residual Convolution block with LeakyReLU.
    Uses FiLM (Feature-wise Linear Modulation) conditioning from speaker style.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        res = x
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.1)
        x = self.norm2(self.conv2(x))
        x = F.leaky_relu(x + res, 0.1)
        return x


class DiffusionProsody(nn.Module):
    """
    Prosody & Style Adaptation Network.
    Modulates text representation using speaker style vector across mel time frames.
    """
    def __init__(self, in_channels=512, style_dim=256, hidden_channels=512, num_blocks=8):
        super().__init__()
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, hidden_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_channels, hidden_channels * 2),  # Scale & Shift
        )
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList([ProsodyBlock(hidden_channels) for _ in range(num_blocks)])
        self.conv_out = nn.Conv1d(hidden_channels, in_channels, 3, padding=1)
        self.norm_out = nn.LayerNorm(in_channels)

    def forward(self, x, style):
        """
        x:     [B, T_mel, hidden_dim]
        style: [B, style_dim]
        """
        # Style scale and shift
        style_params = self.style_proj(style).unsqueeze(-1)  # [B, 2*H, 1]
        scale, shift = style_params.chunk(2, dim=1)         # [B, H, 1] each

        h = self.conv_in(x.transpose(1, 2))                  # [B, H, T_mel]
        h = h * (1.0 + torch.tanh(scale)) + shift            # FiLM modulation

        for block in self.blocks:
            h = block(h)

        h = self.conv_out(h).transpose(1, 2)                 # [B, T_mel, H]
        out = self.norm_out(x + h)                           # Residual connection
        return out
