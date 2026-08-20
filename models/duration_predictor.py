import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    """
    Predicts duration (in mel frames) for each text token.
    Uses dilated 1D convolutions, LayerNorm, and ReLU for non-negative outputs.
    """
    def __init__(self, hidden_dim=512, filter_channels=256, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, filter_channels, 3, padding=1)
        self.ln1 = nn.LayerNorm(filter_channels)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(filter_channels, filter_channels, 3, padding=1)
        self.ln2 = nn.LayerNorm(filter_channels)
        self.drop2 = nn.Dropout(dropout)

        self.proj = nn.Linear(filter_channels, 1)

    def forward(self, x, mask=None):
        """
        x:    [B, T_text, H]
        mask: [B, T_text] boolean mask where True = PAD
        Returns: [B, T_text] predicted duration in frames
        """
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = self.drop1(self.ln1(h.transpose(1, 2)).transpose(1, 2))
        h = F.relu(self.conv2(h))
        h = self.drop2(self.ln2(h.transpose(1, 2)))
        dur = F.relu(self.proj(h).squeeze(-1))  # [B, T], non-negative

        if mask is not None:
            dur = dur.masked_fill(mask, 0.0)
        return dur
