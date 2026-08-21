"""
TamilTTS Acoustic Model Architecture (AI4Bharat / FastPitch / Kokoro-82M SOTA Standard)
========================================================================================
- TextEncoder: Transformer with padding masking and positional encoding.
- StyleEncoder: Extracts reference speaker voice embedding for zero-shot voice cloning.
- DurationPredictor + Monotonic Alignment Search (MAS): True character duration learning.
- DiffusionProsody: Style and prosody latent modulation.
- AcousticProj + PostNet: Dual Mel-Spectrogram generator (coarse + 5-layer refined).
- FullVocoder: Frozen Universal HiFi-GAN Vocoder for 100% buzz-free waveform synthesis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import TextEncoder
from .style_encoder import StyleEncoder
from .duration_predictor import DurationPredictor
from .diffusion import DiffusionProsody
from .postnet import PostNet
from .vocoder import FullVocoder
from .alignment import monotonic_alignment_search


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
      Text -> TextEncoder -> Monotonic Alignment Search (MAS) -> Duration Loss
           -> length_regulate(MAS durations) -> DiffusionProsody -> AcousticProj -> mel_coarse
           -> PostNet -> mel_refined (Dual Mel Loss)

    Inference:
      Text -> TextEncoder -> DurationPredictor -> length_regulate(Predicted Durations)
           -> DiffusionProsody -> AcousticProj -> PostNet -> mel_refined
           -> Pre-trained HiFi-GAN Vocoder -> Clean 22.05kHz Audio
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

        # Kokoro / Tacotron 2 PostNet for Formant & Harmonic Refinement
        self.postnet = PostNet(
            mel_dim=cfg.mel_channels,
            postnet_dim=512,
            n_layers=5,
            kernel_size=5,
            dropout=0.2,
        )

        # Mel initialization matching typical log-mel distributions (prevents initial collapse)
        with torch.no_grad():
            self.acoustic_proj[-1].bias.fill_(-3.5)
            self.acoustic_proj[-1].weight.mul_(0.1)

        self.mel_align_proj = nn.Linear(cfg.mel_channels, cfg.hidden_dim)
        self.vocoder = FullVocoder(
            in_channels=cfg.mel_channels,
            upsample_initial_channel=512,
        )
        self.max_mel_len = cfg.max_mel_len

    def forward(self, text_tokens, ref_mel, target_mel_len=None, text_mask=None, target_dur=None):
        """
        text_tokens:    [B, T_text]
        ref_mel:        [B, 80, T_mel] (target/reference mel)
        target_mel_len: int (optional target frame count)
        text_mask:      [B, T_text] boolean mask (True = PAD)
        target_dur:     [B, T_text] (explicit durations, if provided)
        
        Returns:
            audio:       [B, T_audio] (synthesized waveform via vocoder)
            mel_refined: [B, mel_len, 80] (final post-PostNet mel spectrogram)
            mel_coarse:  [B, mel_len, 80] (raw decoder mel prediction)
            dur_pred:    [B, T_text] (predicted linear duration in frames)
            log_dur_pred:[B, T_text] (predicted log-durations for loss)
        """
        # 1. Extract speaker style embedding from reference mel
        style = self.style_encoder(ref_mel)                          # [B, 256]

        # 2. Encode text with self-attention and pad masking
        x = self.text_encoder(text_tokens, mask=text_mask)           # [B, T_text, 512]

        # 3. Predict durations for each character
        dur_pred, log_dur_pred = self.duration_predictor(x, mask=text_mask)  # [B, T_text]

        # 4. Determine durations for length regulation:
        if self.training and target_dur is None and ref_mel is not None:
            with torch.no_grad():
                mel_proj = self.mel_align_proj(ref_mel.transpose(1, 2))  # [B, T_mel, 512]
                mas_durations, _ = monotonic_alignment_search(x, mel_proj, text_mask=text_mask)
            durations = mas_durations
            mel_len = target_mel_len if target_mel_len else ref_mel.size(2)
        elif target_dur is not None:
            durations = target_dur
            mel_len = target_mel_len if target_mel_len else (ref_mel.size(2) if ref_mel is not None else self.max_mel_len)
        else:
            durations = dur_pred
            total_dur = int(torch.clamp(torch.round(dur_pred), min=0).sum(dim=1).max().item())
            mel_len = target_mel_len if target_mel_len else max(total_dur, 16)

        # 5. Length Regulation
        x_expanded = length_regulate(x, durations, max_len=mel_len)  # [B, mel_len, 512]

        # 6. Modulate prosody & style
        latents = self.diffusion_prosody(x_expanded, style)          # [B, mel_len, 512]

        # 7. Predict Coarse & Refined 80-channel Mel Spectrogram
        mel_coarse = self.acoustic_proj(latents)                     # [B, mel_len, 80]
        mel_residual = self.postnet(mel_coarse)                      # [B, mel_len, 80]
        mel_refined = mel_coarse + mel_residual                      # [B, mel_len, 80]

        # 8. Synthesize waveform via Vocoder
        audio = self.vocoder(mel_refined)                            # [B, T_audio]

        return audio, mel_refined, mel_coarse, dur_pred, log_dur_pred
