import torch
import torch.nn as nn
from .text_encoder import TextEncoder
from .style_encoder import StyleEncoder
from .duration_predictor import DurationPredictor
from .diffusion import DiffusionProsody
from .vocoder import FullVocoder

class TamilTTS(nn.Module):
    """
    Complete TamilTTS model — 68.55M parameters.
    Combines: TextEncoder + StyleEncoder + DurationPredictor + DiffusionProsody + Vocoder
    """
    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=cfg.vocab_size,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.text_encoder_layers,
            num_heads=cfg.text_encoder_heads,
        )
        self.style_encoder = StyleEncoder(
            mel_channels=cfg.mel_channels,
            hidden_dim=cfg.hidden_dim,
            style_dim=cfg.style_dim,
        )
        self.duration_predictor = DurationPredictor(
            hidden_dim=cfg.hidden_dim,
            filter_channels=cfg.duration_filter_channels,
        )
        self.diffusion_prosody = DiffusionProsody(
            in_channels=cfg.hidden_dim,
            style_dim=cfg.style_dim,
            hidden_channels=cfg.hidden_dim,
        )
        self.acoustic_proj = nn.Linear(cfg.hidden_dim, cfg.mel_channels)
        self.vocoder = FullVocoder(
            in_channels=cfg.mel_channels,
            upsample_initial_channel=cfg.vocoder_initial_channels,
        )

    def forward(self, text_tokens, ref_mel, text_mask=None):
        style = self.style_encoder(ref_mel)
        x = self.text_encoder(text_tokens, mask=text_mask)
        dur = self.duration_predictor(x)
        latents = self.diffusion_prosody(x, style)
        mel_pred = self.acoustic_proj(latents)
        audio = self.vocoder(mel_pred)
        return audio, mel_pred, dur
