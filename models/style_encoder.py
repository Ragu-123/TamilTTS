import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleEncoder(nn.Module):
    """
    Reference-mel style encoder: Conv1d stack (stride downsampling) +
    masked attention pooling -> normalized style vector.
    """
    def __init__(self, mel_channels=80, hidden_dim=512, style_dim=256):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(mel_channels, hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2),
        ])
        self.score = nn.Linear(hidden_dim, 1)
        self.out = nn.Linear(hidden_dim, style_dim)

    def forward(self, mel, mel_lens=None):
        """
        mel:     [B, 80, T]
        mel_lens: [B] frame lengths or None
        Returns [B, style_dim] (L2-normalized).
        """
        h = mel
        for conv in self.convs:
            h = F.relu(conv(h))

        if mel_lens is not None:
            reduced_lens = torch.ceil(mel_lens.float() / 4.0).long().clamp(min=1)
            idx = torch.arange(h.size(2), device=h.device).unsqueeze(0)
            pad_mask = idx >= reduced_lens.unsqueeze(1)  # [B, T'] True=pad
            scores = self.score(h.transpose(1, 2)).squeeze(-1)  # [B, T']
            scores = scores.masked_fill(pad_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)             # [B, T']
            pooled = torch.bmm(weights.unsqueeze(1), h.transpose(1, 2)).squeeze(1)  # [B, H]
        else:
            scores = self.score(h.transpose(1, 2)).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            pooled = torch.bmm(weights.unsqueeze(1), h.transpose(1, 2)).squeeze(1)

        return F.normalize(self.out(pooled), dim=-1)
