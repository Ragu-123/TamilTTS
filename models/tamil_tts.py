import torch
import torch.nn as nn
from .text_encoder import TextEncoder
from .style_encoder import StyleEncoder
from .duration_predictor import DurationPredictor
from .diffusion import DiffusionProsody
from .vocoder import FullVocoder

class TamilTTS(nn.Module):
    """
    Complete TamilTTS — 68.55M parameters.
    Returns: (audio_waveform, mel_prediction, duration_prediction)
    """
    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=cfg.vocab_size, hidden_dim=cfg.hidden_dim,
            num_layers=cfg.text_encoder_layers, num_heads=cfg.text_encoder_heads,
        )
        self.style_encoder = StyleEncoder(
            mel_channels=cfg.mel_channels, hidden_dim=cfg.hidden_dim, style_dim=cfg.style_dim,
        )
        self.duration_predictor = DurationPredictor(
            hidden_dim=cfg.hidden_dim, filter_channels=cfg.duration_filter_channels,
        )
        self.diffusion_prosody = DiffusionProsody(
            in_channels=cfg.hidden_dim, style_dim=cfg.style_dim, hidden_channels=cfg.hidden_dim,
        )
        self.acoustic_proj = nn.Linear(cfg.hidden_dim, cfg.mel_channels)
        self.vocoder = FullVocoder(
            in_channels=cfg.mel_channels, upsample_initial_channel=cfg.vocoder_initial_channels,
        )

    def forward(self, text_tokens, ref_mel, text_mask=None):
        style    = self.style_encoder(ref_mel)              # [B, 256]
        x        = self.text_encoder(text_tokens, text_mask) # [B, T_text, 512]
        dur      = self.duration_predictor(x)                # [B, T_text]
        latents  = self.diffusion_prosody(x, style)          # [B, T_text, 512]
        mel_pred = self.acoustic_proj(latents)               # [B, T_text, 80]
        audio    = self.vocoder(mel_pred)                    # [B, T_audio]
        return audio, mel_pred, dur
