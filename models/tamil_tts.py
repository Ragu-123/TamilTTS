"""
TamilTTS Acoustic Model Architecture (FastPitch / RAD-TTS SOTA Standard)
=========================================================================
- TextEncoder: Transformer with Sinusoidal Positional Encoding & Pad Masking.
- StyleEncoder: Extracts reference speaker voice embedding with 50% training style dropout (Anti-Leakage).
- AlignmentNetwork: Exact RAD-TTS Forward-Sum + True Binarization Loss + Viterbi MAS on unified log_A.
- DurationPredictor: Dilated Conv1D log-duration predictor trained on alignment-derived durations.
- DiffusionProsody: FiLM-modulated prosody & style latent modulation.
- AcousticProj + PostNet: Dual Mel-Spectrogram generator (coarse + 5-layer refined at 22.05 kHz).
- FullVocoder: Frozen Universal HiFi-GAN Vocoder (13.93M parameters, 22.05 kHz output).
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
from .alignment import AlignmentNetwork


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
        # Default learned speaker embedding (used when no reference voice is provided / style dropout)
        self.default_style = nn.Parameter(torch.randn(1, cfg.style_dim) * 0.02)

        self.aligner = AlignmentNetwork(
            text_dim=cfg.hidden_dim,
            mel_dim=cfg.mel_channels,
            attn_dim=getattr(cfg, "aligner_dim", 128),
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

        # Kokoro / Tacotron 2 5-layer PostNet for Formant & Harmonic Refinement
        self.postnet = PostNet(
            mel_dim=cfg.mel_channels,
            postnet_dim=512,
            n_layers=5,
            kernel_size=5,
            dropout=0.1,
        )

        # Mel initialization matching natural log-mel distributions (mean ~ -6.5)
        nn.init.normal_(self.acoustic_proj[-1].weight, std=0.02)
        nn.init.constant_(self.acoustic_proj[-1].bias, -6.5)

        self.vocoder = FullVocoder(
            in_channels=cfg.mel_channels,
            upsample_initial_channel=512,
        )

    def forward(self, text_tokens, text_lens, ref_mel=None, mel_lens=None, target_dur=None, style_dropout_p=0.5, return_audio=False):
        """
        text_tokens: [B, T_text]
        text_lens:   [B] (actual token lengths)
        ref_mel:     [B, 80, T_mel] (optional during inference)
        mel_lens:    [B] (actual mel frame lengths)
        target_dur:  [B, T_text] (explicit durations, if provided)

        Returns:
            audio:            [B, T_audio] or None
            mel_refined:      [B, mel_len, 80] (post-PostNet mel spectrogram)
            mel_coarse:       [B, mel_len, 80] (pre-PostNet mel prediction)
            dur_pred:         [B, T_text] (predicted linear duration)
            log_dur_pred:     [B, T_text] (predicted log-duration)
            align_dur:        [B, T_text] (alignment-derived duration targets)
            forward_sum_loss: scalar tensor (from RAD-TTS aligner)
            bin_loss:         scalar tensor (from RAD-TTS aligner)
        """
        B, T_text = text_tokens.shape
        device = text_tokens.device

        text_mask = torch.arange(T_text, device=device).unsqueeze(0) >= text_lens.unsqueeze(1)  # True = PAD

        # 1. Speaker Style Conditioning (with Anti-Leakage Style Dropout during training)
        default_style_exp = self.default_style.expand(B, -1)
        if ref_mel is not None:
            # Detach ref_mel when extracting style to prevent backprop gradient shortcuts
            style_extracted = self.style_encoder(ref_mel.detach())
            if self.training and style_dropout_p > 0.0:
                # 50% style dropout: forces TextEncoder to learn prosody independently
                drop_mask = (torch.rand(B, 1, device=device) < style_dropout_p).float()
                style = (1.0 - drop_mask) * style_extracted + drop_mask * default_style_exp
            else:
                style = style_extracted
        else:
            style = default_style_exp

        # 2. Encode text with self-attention, positional encoding, and pad masking
        x = self.text_encoder(text_tokens, mask=text_mask)  # [B, T_text, 512]

        # 3. Predict durations for each character
        dur_pred, log_dur_pred = self.duration_predictor(x, mask=text_mask)  # [B, T_text]

        # 4. Alignment & Duration Assignment:
        forward_sum_loss = torch.tensor(0.0, device=device)
        bin_loss = torch.tensor(0.0, device=device)

        if target_dur is not None:
            # Kokoro-style direct ground-truth duration supervision
            align_dur = target_dur
            durations = target_dur
            mel_len = int(mel_lens.max().item()) if mel_lens is not None else max(int(durations.sum(dim=1).max().item()), 16)
        elif self.training and ref_mel is not None and mel_lens is not None:
            # RAD-TTS Alignment Network (Forward-Sum + True Binarization on same log_A)
            align_dur, hard_path, forward_sum_loss, bin_loss = self.aligner(x, ref_mel, text_lens, mel_lens)
            durations = align_dur
            mel_len = int(mel_lens.max().item())
        else:
            align_dur = dur_pred
            dur_rounded = torch.clamp(torch.round(dur_pred), min=1.0) * (~text_mask).float()
            durations = dur_rounded
            total_dur = int(dur_rounded.sum(dim=1).max().item())
            mel_len = max(total_dur, 16)

        # 5. Length Regulation
        x_expanded = length_regulate(x, durations, max_len=mel_len)  # [B, mel_len, 512]

        # 6. Modulate prosody & style
        latents = self.diffusion_prosody(x_expanded, style)          # [B, mel_len, 512]

        # 7. Predict Coarse & Refined 80-channel Mel Spectrogram (22.05 kHz)
        mel_coarse = self.acoustic_proj(latents)                     # [B, mel_len, 80]
        mel_residual = self.postnet(mel_coarse)                      # [B, mel_len, 80]
        mel_refined = mel_coarse + mel_residual                      # [B, mel_len, 80]

        # 8. Synthesize waveform via Vocoder ONLY when required (inference/eval/SLM)
        if return_audio or not self.training:
            audio = self.vocoder(mel_refined)                        # [B, T_audio]
        else:
            audio = None

        return audio, mel_refined, mel_coarse, dur_pred, log_dur_pred, align_dur, forward_sum_loss, bin_loss
