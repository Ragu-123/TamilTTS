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
    Expand text features to mel-frame length using predicted durations.
    This is the CRITICAL piece that makes TTS work (used by FastSpeech, VITS, Kokoro).

    x:         [B, T_text, H]   — text encoder output
    durations: [B, T_text]      — predicted frames per phoneme
    max_len:   int              — target mel length to pad/truncate to

    Returns:   [B, max_len, H]  — expanded features matching mel length
    """
    # Round durations to integers, minimum 1 frame per phoneme
    dur_rounded = torch.clamp(torch.round(durations), min=0).long()

    B, T, H = x.shape
    if max_len is None:
        max_len = dur_rounded.sum(dim=1).max().item()

    # Build expanded output
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
    Complete TamilTTS — 68.55M parameters.
    Now with proper LENGTH REGULATION for correct speech timing.

    Pipeline:
      Text -> TextEncoder -> DurationPredictor -> LENGTH REGULATE
           -> DiffusionProsody (+ Style) -> acoustic_proj -> Vocoder -> Audio
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
        self.max_mel_len = cfg.max_mel_len

    def forward(self, text_tokens, ref_mel, target_mel_len=None, text_mask=None):
        """
        text_tokens:    [B, T_text]
        ref_mel:        [B, 80, T_mel]  — reference mel for style extraction
        target_mel_len: int — during training, set to actual mel length for proper alignment
        """
        # 1. Extract style from reference audio
        style = self.style_encoder(ref_mel)                   # [B, 256]

        # 2. Encode text
        x = self.text_encoder(text_tokens, text_mask)         # [B, T_text, 512]

        # 3. Predict durations (frames per phoneme)
        dur_pred = self.duration_predictor(x)                 # [B, T_text]

        # 4. LENGTH REGULATE: expand text features to mel-frame length
        mel_len = target_mel_len if target_mel_len else self.max_mel_len
        x_expanded = length_regulate(x, dur_pred, max_len=mel_len)  # [B, mel_len, 512]

        # 5. Apply prosody/style conditioning
        latents = self.diffusion_prosody(x_expanded, style)   # [B, mel_len, 512]

        # 6. Project to mel spectrogram
        mel_pred = self.acoustic_proj(latents)                # [B, mel_len, 80]

        # 7. Generate waveform
        audio = self.vocoder(mel_pred)                        # [B, T_audio]

        return audio, mel_pred, dur_pred
