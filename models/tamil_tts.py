import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import TextEncoder
from .style_encoder import StyleEncoder
from .duration_predictor import DurationPredictor
from .diffusion import DiffusionProsody
from .vocoder import FullVocoder


def length_regulate(x, durations, max_len=None):
    """
    Expand text representations to mel-frame length using duration alignments.

    x:         [B, T_text, H]   — text encoder output
    durations: [B, T_text]      — frame durations per character
    max_len:   int or None      — target mel length

    Returns:   [B, max_len, H]  — time-expanded acoustic hidden states
    """
    dur_rounded = torch.clamp(torch.round(durations), min=0).long()
    B, T, H = x.shape

    if max_len is None:
        max_len = max(int(dur_rounded.sum(dim=1).max().item()), 1)
    else:
        max_len = max(int(max_len), 1)

    output = torch.zeros(B, max_len, H, device=x.device, dtype=x.dtype)
    for b in range(B):
        pos = 0
        for t in range(T):
            d = dur_rounded[b, t].item()
            if d > 0 and pos < max_len:
                end = min(pos + d, max_len)
                output[b, pos:end, :] = x[b, t, :]
                pos = end
            if pos >= max_len:
                break
    return output


class TamilTTS(nn.Module):
    """
    Complete End-to-End TamilTTS Architecture.
    
    Training:
      Text -> TextEncoder (masked) -> DurationPredictor -> length_regulate(using TARGET_DUR)
           -> Prosody/Style Diffusion -> Acoustic Proj -> Vocoder (HiFi-GAN MRF) -> Audio

    Inference:
      Text -> TextEncoder -> DurationPredictor -> length_regulate(using PREDICTED_DUR)
           -> Prosody/Style Diffusion -> Acoustic Proj -> Vocoder -> Clean Speech
    """
    def __init__(self, cfg):
        super().__init__()
        self.text_encoder = TextEncoder(
            vocab_size=getattr(cfg, "vocab_size", 256),
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
            num_blocks=cfg.diffusion_blocks,
        )
        self.acoustic_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(cfg.hidden_dim // 2, cfg.mel_channels),
        )
        self.vocoder = FullVocoder(
            in_channels=cfg.mel_channels,
            upsample_initial_channel=512,
        )
        self.max_mel_len = cfg.max_mel_len

    def forward(self, text_tokens, ref_mel, target_mel_len=None, text_mask=None, target_dur=None):
        """
        text_tokens:    [B, T_text]
        ref_mel:        [B, 80, T_mel] (reference for style)
        target_mel_len: int (optional target frame count)
        text_mask:      [B, T_text] boolean mask (True = PAD)
        target_dur:     [B, T_text] (ground-truth duration during training)
        """
        # 1. Extract speaker style embedding from reference mel
        style = self.style_encoder(ref_mel)                          # [B, 256]

        # 2. Encode text with self-attention and pad masking
        x = self.text_encoder(text_tokens, mask=text_mask)           # [B, T_text, 512]

        # 3. Predict durations for each character
        dur_pred = self.duration_predictor(x, mask=text_mask)        # [B, T_text]

        # 4. Length Regulation
        # During TRAINING: use target_dur for rock-solid phoneme-to-acoustic alignment
        # During INFERENCE: use model's predicted dur_pred
        durations = target_dur if target_dur is not None else dur_pred
        mel_len = target_mel_len if target_mel_len else self.max_mel_len
        x_expanded = length_regulate(x, durations, max_len=mel_len)  # [B, mel_len, 512]

        # 5. Modulate prosody & style
        latents = self.diffusion_prosody(x_expanded, style)          # [B, mel_len, 512]

        # 6. Predict mel spectrogram
        mel_pred = self.acoustic_proj(latents)                       # [B, mel_len, 80]

        # 7. Synthesize audio waveform with HiFi-GAN MRF vocoder
        audio = self.vocoder(mel_pred)                               # [B, T_audio]

        return audio, mel_pred, dur_pred
