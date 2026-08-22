import math

import torch
import torch.nn as nn


class VariancePredictor(nn.Module):
    """
    Standard duration/variance predictor:
    two blocks of [Conv1d(k3) -> LN -> GELU -> Dropout] then Linear -> scalar.
    """
    def __init__(self, hidden_dim, filter_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv_stack = nn.ModuleList()
        in_ch = hidden_dim
        for _ in range(2):
            self.conv_stack.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, filter_channels, kernel_size,
                              padding=(kernel_size - 1) // 2),
                    nn.LayerNorm(filter_channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            in_ch = filter_channels
        self.proj = nn.Linear(filter_channels, 1)

    def forward(self, x, mask_true_is_pad=None):
        """
        x: [B, T, H]
        Returns raw predictions [B, T], pad positions filled with 0.0.
        """
        y = x.transpose(1, 2)
        for layer in self.conv_stack:
            conv_out = layer[0](y)                    # Conv1d -> [B, F, T]
            ln_out = layer[1](conv_out.transpose(1, 2)).transpose(1, 2)
            act = layer[2](ln_out)
            y = layer[3](act)
        out = self.proj(y.transpose(1, 2)).squeeze(-1)  # [B, T]

        if mask_true_is_pad is not None:
            out = out.masked_fill(mask_true_is_pad, 0.0)
        return out


class DurationHead(VariancePredictor):
    """
    Duration predictor with log-mean init: final Linear bias=log(5),
    weight zeros (so initial predicted durations ~= exp(log 5) = 5 frames).
    """
    def __init__(self, hidden_dim, filter_channels, kernel_size=3, dropout=0.1):
        super().__init__(hidden_dim, filter_channels, kernel_size=kernel_size, dropout=dropout)
        nn.init.zeros_(self.proj.weight)
        with torch.no_grad():
            self.proj.bias.fill_(math.log(5.0))


class PitchHead(VariancePredictor):
    pass


class EnergyHead(VariancePredictor):
    pass


class PitchEmbedder(nn.Module):
    """Embed log-F0 contour into hidden space via small Conv1d stack."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

    def forward(self, logf0):
        """
        logf0: [B, T] -> [B, T, hidden_dim]
        """
        return self.net(logf0.unsqueeze(1)).transpose(1, 2)
