import torch
import torch.nn as nn

class DurationPredictor(nn.Module):
    """Predicts phoneme durations — ~0.59M params."""
    def __init__(self, hidden_dim=512, filter_channels=256):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, filter_channels, 3, padding=1)
        self.ln1 = nn.LayerNorm(filter_channels)
        self.conv2 = nn.Conv1d(filter_channels, filter_channels, 3, padding=1)
        self.ln2 = nn.LayerNorm(filter_channels)
        self.proj = nn.Linear(filter_channels, 1)

    def forward(self, x):
        # x: [B, T, H]
        x = x.transpose(1, 2)                  # [B, H, T]
        x = torch.relu(self.conv1(x))          # [B, F, T]
        x = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        x = torch.relu(self.conv2(x))
        x = self.ln2(x.transpose(1, 2))        # [B, T, F]
        return self.proj(x).squeeze(-1)         # [B, T]
