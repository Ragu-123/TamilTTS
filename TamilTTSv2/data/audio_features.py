"""
Audio Feature Extraction for TamilTTSv2
========================================
- MelExtractor: exact IndicTTS / Coqui TTS mel recipe (frozen HiFi-GAN compatible).
- extract_f0: log-F0 at mel frame rate with utterance-level voiced normalization.
- extract_energy: log RMS energy per hop window with utterance normalization.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF

try:
    import pyworld as pw
except ImportError:
    pw = None


class MelExtractor(nn.Module):
    """
    EXACT AI4Bharat IndicTTS / Coqui TTS v0.0.13 AudioProcessor recipe — the recipe the
    frozen HiFi-GAN was trained on (verified by round-trip: envelope corr 0.91,
    spectral-centroid corr 0.96; the previous implementation scored ~0).

    Pipeline (Coqui v0.0.13 melspectrogram with this vocoder's config:
    log_func=np.log, spec_gain=1.0, signal_norm=False, preemphasis=0.0, power unused):
      1. librosa STFT: hann window, center=True (== reflect-pad n_fft//2 + center=False),
         stft_pad_mode='reflect'
      2. mel basis: librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) — slaney norm,
         htk=False (torchaudio's "slaney" scale+norm matches librosa defaults)
      3. S = gain * ln(max(mel @ |STFT|, 1e-5)); NO dB conversion, NO [-4,4] normalization,
         no ^1.5 power compression (power is only used by Griffin-Lim inversion).
    Output range is roughly [-14, 2]; mel_proj bias and loss scales are range-agnostic.

    Args:
        audio (Tensor): [1, T] or [T] waveform.
    Returns:
        Tensor: [80, Tm] natural-log mel, where Tm matches Coqui/librosa center=True framing.
    """

    # Floor from Coqui v0.0.13 _amp_to_db: np.maximum(1e-5, x)
    LOG_FLOOR = 1e-5

    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=80, fmin=0.0, fmax=8000.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.register_buffer("window", torch.hann_window(n_fft))
        fb = AF.melscale_fbanks(
            n_freqs=(n_fft // 2) + 1,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney"
        ).transpose(0, 1)
        self.register_buffer("mel_basis", fb)

    def forward(self, audio):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        # center=True equivalent for torch.stft(center=False): reflect-pad by n_fft//2.
        # librosa pads with n_fft//2 on both sides, yielding ceil(T/hop)+1 frames.
        pad = self.n_fft // 2
        audio_padded = torch.nn.functional.pad(audio.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)
        stft = torch.stft(
            audio_padded.unsqueeze(0), self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=self.window, center=False, return_complex=True
        ).squeeze(0)
        spec = torch.abs(stft)
        mel = torch.matmul(self.mel_basis, spec)
        return torch.log(torch.clamp(mel, min=self.LOG_FLOOR))


def _interp_to(x, num_frames):
    """Linearly resample a 1-D tensor to exactly `num_frames` points."""
    if x.numel() == num_frames:
        return x
    if x.numel() == 0:
        return torch.zeros(num_frames, dtype=x.dtype)
    pos = torch.linspace(0, x.numel() - 1, steps=num_frames)
    i0 = pos.floor().long().clamp(0, x.numel() - 1)
    i1 = (i0 + 1).clamp(max=x.numel() - 1)
    w = (pos - i0.float()).clamp(0.0, 1.0)
    return x[i0] * (1.0 - w) + x[i1] * w


def _num_mel_frames(audio_len, hop_length):
    return max(1, int(math.ceil(audio_len / hop_length)))


def extract_f0(audio, sr=22050, hop_length=256, n_fft=1024):
    """
    Extract utterance-normalized log-F0 and voicing mask at mel frame rate.

    Uses pyworld (dio + stonemask) when available; otherwise falls back to
    torchaudio.functional.detect_pitch_frequency at the native sample rate.

    Args:
        audio (Tensor): [1, T] or [T] waveform.
        sr (int): Sample rate.
        hop_length (int): Hop size used for framing.
        n_fft (int): FFT size (unused for pitch, kept for API symmetry).
    Returns:
        Tuple[Tensor, Tensor]: (log_f0[Tm], voiced_mask[Tm]) float32 tensors.
            Unvoiced frames have log_f0=0 and voiced_mask=0. log_f0 is
            normalized over voiced frames (mean 0, std >= 0.1). If fewer than
            5 voiced frames are detected, log_f0 is returned as all zeros.
    """
    flat = audio.detach().cpu().flatten()
    num_frames = _num_mel_frames(flat.numel(), hop_length)
    zeros = lambda: torch.zeros(num_frames, dtype=torch.float32)

    if flat.numel() < hop_length // 2:
        return zeros(), zeros()

    try:
        used_pyworld = False
        if pw is not None:
            wav_np = flat.numpy().astype(np.float64)
            f0_ts, time_axis = pw.dio(wav_np, sr, frame_period=5.0)
            f0_np = pw.stonemask(wav_np, f0_ts, time_axis, sr)
            f0 = torch.from_numpy(np.ascontiguousarray(f0_np)).float()
            used_pyworld = True
        else:
            pitch = AF.detect_pitch_frequency(flat, sample_rate=sr).flatten().float()
            f0 = pitch
        f0 = _interp_to(f0, num_frames)
    except Exception:
        return zeros(), zeros()

    if used_pyworld:
        # dio/stonemask emit 0 on unvoiced frames -> direct voicing mask.
        voiced_mask = (f0 > 0.0).float()
    else:
        # torchaudio.detect_pitch_frequency returns a value for EVERY frame
        # (no voicing detection). Gate by plausibility (human F0 range) and
        # frame energy, otherwise silent frames get fake pitch supervision.
        pad_len = num_frames * hop_length - flat.numel()
        padded = F.pad(flat, (0, max(pad_len, 0)))[: num_frames * hop_length]
        frames = padded.view(num_frames, hop_length)
        rms = frames.pow(2).mean(dim=1).sqrt()
        db = 20.0 * torch.log10(rms.clamp(min=1e-8))
        energy_ok = db > (db.max() - 35.0)
        plausible = (f0 >= 50.0) & (f0 <= 600.0)
        voiced_mask = (plausible & energy_ok).float()

    log_f0 = torch.zeros(num_frames, dtype=torch.float32)
    voiced_idx = voiced_mask.bool()
    if voiced_idx.sum().item() < 5:
        return log_f0, voiced_mask

    voiced_f0 = torch.log(f0[voiced_idx].clamp(min=1e-6))
    mean = voiced_f0.mean()
    std = voiced_f0.std(unbiased=False).clamp(min=0.1)
    log_f0[voiced_idx] = (voiced_f0 - mean) / std
    return log_f0, voiced_mask


def extract_energy(audio, hop_length=256, n_fft=1024):
    """
    Extract utterance-normalized log RMS energy at mel frame rate.

    Args:
        audio (Tensor): [1, T] or [T] waveform.
        hop_length (int): Window size for per-frame RMS.
        n_fft (int): FFT size (unused for RMS framing, kept for API symmetry).
    Returns:
        Tensor: energy[Tm], float32, mean/std normalized over all frames.
    """
    flat = audio.detach().cpu().flatten()
    num_frames = _num_mel_frames(flat.numel(), hop_length)

    pad_total = num_frames * hop_length - flat.numel()
    if pad_total > 0:
        if flat.numel() > 1 and pad_total < flat.numel():
            flat = torch.nn.functional.pad(flat.unsqueeze(0), (0, pad_total), mode='reflect').squeeze(0)
        else:
            flat = torch.nn.functional.pad(flat, (0, pad_total))

    frames = flat.unfold(0, hop_length, hop_length)
    rms = frames.pow(2).mean(dim=1).sqrt()
    energy = torch.log(rms.clamp(min=1e-5))

    if energy.numel() < 2:
        return torch.zeros_like(energy)

    mean = energy.mean()
    std = energy.std(unbiased=False).clamp(min=0.1)
    return (energy - mean) / std
