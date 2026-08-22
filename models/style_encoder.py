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

    def forward(self, mel, mel_lens=None):
        x = self.convs(mel).transpose(1, 2)  # [B, T//4, hidden_dim]
        if mel_lens is not None and mel_lens.dim() > 0:
            reduced_lens = torch.clamp((mel_lens + 3) // 4, min=1)
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x, reduced_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h = self.rnn(packed)
            return self.proj(h[-1])
        else:
            _, h = self.rnn(x)
            return self.proj(h[-1])
