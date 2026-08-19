import torch
import torch.nn as nn

class StyleEncoder(nn.Module):
    """Conv + GRU style extractor, ~4.54M params."""
    def __init__(self, mel_channels=80, hidden_dim=512, style_dim=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(mel_channels, hidden_dim, 5, padding=2), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, style_dim)

    def forward(self, mel):
        # mel: [B, 80, T_mel]
        x = self.convs(mel).transpose(1, 2)   # [B, T', 512]
        _, h = self.rnn(x)                     # h: [1, B, 512]
        return self.proj(h[-1])                # [B, 256]
