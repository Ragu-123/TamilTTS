import torch
import torch.nn as nn
import torch.nn.functional as F


class DurationPredictor(nn.Module):
    """
    Predicts log-duration for each text token.
    Uses dilated 1D convolutions, LayerNorm, and linear projection.
    
    Predicting in log-space (Kokoro-82M / FastPitch standard) ensures:
    1. Gradients never saturate or hit zero.
    2. Relative syllable errors are penalized proportionally.
    3. Outputs are always mathematically positive after torch.exp().
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
        Returns:
            dur:     [B, T_text] predicted duration in frames (>= 0)
            log_dur: [B, T_text] raw log-scale duration predictions
        """
        h = x.transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = self.drop1(self.ln1(h.transpose(1, 2)).transpose(1, 2))
        h = F.relu(self.conv2(h))
        h = self.drop2(self.ln2(h.transpose(1, 2)))
        log_dur = self.proj(h).squeeze(-1)  # [B, T]

        # Convert to linear frame duration
        dur = torch.exp(log_dur).clamp(min=0.0, max=100.0)

        if mask is not None:
            log_dur = log_dur.masked_fill(mask, -10.0)
            dur = dur.masked_fill(mask, 0.0)

        return dur, log_dur
