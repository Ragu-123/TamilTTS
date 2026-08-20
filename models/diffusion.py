import torch
import torch.nn as nn

class ResBlock1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(torch.relu(self.conv1(x)))

class DiffusionProsody(nn.Module):
    """8-block residual network conditioned on style, ~14.3M params."""
    def __init__(self, in_channels=512, style_dim=256, hidden_channels=512):
        super().__init__()
        self.style_cond = nn.Linear(style_dim, hidden_channels)
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList([ResBlock1d(hidden_channels) for _ in range(8)])
        self.conv_out = nn.Conv1d(hidden_channels, in_channels, 3, padding=1)

    def forward(self, x, style):
        s = self.style_cond(style).unsqueeze(-1)
        x = self.conv_in(x.transpose(1, 2)) + s
        for b in self.blocks:
            x = b(x)
        return self.conv_out(x).transpose(1, 2)
