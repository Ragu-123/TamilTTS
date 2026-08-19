import torch
import torch.nn as nn

class TextEncoder(nn.Module):
    """10-layer Transformer, 512 hidden, ~31.6M params."""
    def __init__(self, vocab_size=128, hidden_dim=512, num_layers=10, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        return self.encoder(x, src_key_padding_mask=mask)
