import torch
import torch.nn as nn
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self, vocab_size=128, hidden_dim=512, num_layers=6, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.phoneme_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        out = self.encoder(x, src_key_padding_mask=mask)
        return self.phoneme_proj(out)

class StyleEncoder(nn.Module):
    def __init__(self, mel_channels=80, style_dim=128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(mel_channels, 256, kernel_size=5, padding=2), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Conv1d(512, 1024, kernel_size=5, stride=2, padding=2), nn.ReLU(), nn.BatchNorm1d(1024),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(1024, style_dim)
        
    def forward(self, mel):
        x = self.convs(mel)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

class DiffusionProsody(nn.Module):
    def __init__(self, hidden_dim=512, style_dim=128):
        super().__init__()
        self.time_embed = nn.Linear(1, hidden_dim)
        self.style_proj = nn.Linear(style_dim, hidden_dim)
        self.net = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1), nn.GELU(), nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1))
        self.proj_out = nn.Conv1d(hidden_dim, 2, 1)
        
    def forward(self, x, style_vector, diffusion_time):
        x = x.transpose(1, 2)
        t_emb = self.time_embed(diffusion_time.unsqueeze(-1)).unsqueeze(-1)
        s_emb = self.style_proj(style_vector).unsqueeze(-1)
        out = self.proj_out(self.net(x + t_emb + s_emb))
        return out.transpose(1, 2)

class HiFiGANVocoder(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.pre_conv = nn.Conv1d(in_channels, 512, kernel_size=7, padding=3)
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose1d(512, 256, 16, 8, 4), nn.ConvTranspose1d(256, 128, 16, 8, 4),
            nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.ConvTranspose1d(64, 32, 4, 2, 1),
        ])
        self.post_conv = nn.Conv1d(32, 1, 7, padding=3)
        
    def forward(self, x):
        x = F.leaky_relu(self.pre_conv(x.transpose(1, 2)), 0.1)
        for up in self.upsamples: x = F.leaky_relu(up(x), 0.1)
        return torch.tanh(self.post_conv(x)).squeeze(1)

class TamilTTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.style_encoder = StyleEncoder()
        self.diffusion_prosody = DiffusionProsody()
        self.vocoder = HiFiGANVocoder()
        
    def forward(self, text_tokens, ref_mel, diffusion_time):
        text_features = self.text_encoder(text_tokens)
        style_vector = self.style_encoder(ref_mel)
        prosody = self.diffusion_prosody(text_features, style_vector, diffusion_time)
        audio = self.vocoder(text_features)
        return audio, prosody
