"""
TamilTTS Acoustic Model Architecture (AI4Bharat / FastPitch / Kokoro SOTA Standard)
====================================================================================
- TextEncoder: Transformer with Sinusoidal Positional Encoding & Pad Masking.
- StyleEncoder: Extracts reference speaker voice embedding for zero-shot voice cloning.
- AlignmentModule: CTC-supervised Conv1D alignment head for mathematically grounded text-to-mel alignment.
- DurationPredictor: Learns log-durations from CTC alignment targets.
- DiffusionProsody: FiLM-modulated style and prosody latent adaptation.
- AcousticProj + PostNet: Dual Mel-Spectrogram generator (coarse + 5-layer refined at 22.05 kHz).
- FullVocoder: Frozen Universal HiFi-GAN Vocoder (22.05 kHz output).
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
from .alignment import AlignmentModule, extract_alignment_durations


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
    """
    def __init__(self, cfg):
        super().__init__()
        self.vocab_size = getattr(cfg, "vocab_size", 256)
        self.text_encoder = TextEncoder(
            vocab_size=self.vocab_size,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.text_encoder_layers,
            num_heads=cfg.text_encoder_heads,
        )
        self.style_encoder = StyleEncoder(
            mel_channels=cfg.mel_channels,
            hidden_dim=cfg.hidden_dim,
            style_dim=cfg.style_dim,
        )
        self.aligner = AlignmentModule(
            mel_channels=cfg.mel_channels,
            hidden_dim=256,
            vocab_size=self.vocab_size,
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
            dropout=0.5,
        )

        # Mel initialization matching natural log-mel distributions (mean ~ -3.5)
        with torch.no_grad():
            self.acoustic_proj[-1].bias.fill_(-3.5)
            self.acoustic_proj[-1].weight.mul_(0.1)

        self.vocoder = FullVocoder(
            in_channels=cfg.mel_channels,
            upsample_initial_channel=512,
        )
        self.max_mel_len = cfg.max_mel_len

    def forward(self, text_tokens, ref_mel, target_mel_len=None, text_mask=None, target_dur=None):
        """
        text_tokens:    [B, T_text]
        ref_mel:        [B, 80, T_mel]
        target_mel_len: int (optional target frame count)
        text_mask:      [B, T_text] boolean mask (True = PAD)
        target_dur:     [B, T_text] (explicit durations, if provided)
        
        Returns:
            audio:        [B, T_audio] (synthesized waveform via vocoder)
            mel_refined:  [B, mel_len, 80] (final post-PostNet mel spectrogram)
            mel_coarse:   [B, mel_len, 80] (raw decoder mel prediction)
            dur_pred:     [B, T_text] (predicted linear duration in frames)
            log_dur_pred: [B, T_text] (predicted log-durations for loss)
            ctc_log_probs:[T_mel, B, vocab_size] (for CTC alignment loss)
            align_dur:    [B, T_text] (ground-truth duration from alignment)
        """
        # 1. Extract speaker style embedding from reference mel
        style = self.style_encoder(ref_mel)                          # [B, 256]

        # 2. Encode text with self-attention, positional encoding, and pad masking
        x = self.text_encoder(text_tokens, mask=text_mask)           # [B, T_text, 512]

        # 3. Supervised CTC Alignment
        ctc_logits, ctc_log_probs = self.aligner(ref_mel)            # logits: [B, vocab, T_mel], log_probs: [T_mel, B, vocab]

        # 4. Predict durations for each character
        dur_pred, log_dur_pred = self.duration_predictor(x, mask=text_mask)  # [B, T_text]

        # 5. Determine durations for length regulation:
        if self.training and target_dur is None and ref_mel is not None:
            align_dur, _ = extract_alignment_durations(text_tokens, ctc_logits, text_mask=text_mask)
            durations = align_dur
            mel_len = target_mel_len if target_mel_len else ref_mel.size(2)
        elif target_dur is not None:
            align_dur = target_dur
            durations = target_dur
            mel_len = target_mel_len if target_mel_len else (ref_mel.size(2) if ref_mel is not None else self.max_mel_len)
        else:
            align_dur = dur_pred
            durations = dur_pred
            total_dur = int(torch.clamp(torch.round(dur_pred), min=0).sum(dim=1).max().item())
            mel_len = target_mel_len if target_mel_len else max(total_dur, 16)

        # 6. Length Regulation
        x_expanded = length_regulate(x, durations, max_len=mel_len)  # [B, mel_len, 512]

        # 7. Modulate prosody & style
        latents = self.diffusion_prosody(x_expanded, style)          # [B, mel_len, 512]

        # 8. Predict Coarse & Refined 80-channel Mel Spectrogram (22.05 kHz)
        mel_coarse = self.acoustic_proj(latents)                     # [B, mel_len, 80]
        mel_residual = self.postnet(mel_coarse)                      # [B, mel_len, 80]
        mel_refined = mel_coarse + mel_residual                      # [B, mel_len, 80]

        # 9. Synthesize waveform via Vocoder
        audio = self.vocoder(mel_refined)                            # [B, T_audio]

        return audio, mel_refined, mel_coarse, dur_pred, log_dur_pred, ctc_log_probs, align_dur
