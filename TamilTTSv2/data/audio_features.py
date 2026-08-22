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
import torchaudio.functional as AF

try:
    import pyworld as pw
except ImportError:
    pw = None


class MelExtractor(nn.Module):
    """
    Exact IndicTTS / Coqui TTS Mel-Spectrogram Processor (100% matched to frozen vocoder).
    22.05 kHz, n_fft=1024, hop=256, 80 mels, slaney scale/norm, fmin=0, fmax=8000,
    ^1.5 power compression, normalized to [-4, 4], reflect padding.

    Args:
        audio (Tensor): [1, T] or [T] waveform.
    Returns:
        Tensor: [80, Tm] normalized log-mel, where Tm = floor((T - hop) / hop) + 1.
    """

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
        pad = int((self.n_fft - self.hop_length) / 2)
        audio_padded = torch.nn.functional.pad(audio.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)
        stft = torch.stft(
            audio_padded, self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
            window=self.window, center=False, return_complex=True
        )
        spec = (torch.abs(stft) + 1e-9) ** 1.5
        mel = torch.matmul(self.mel_basis, spec)
        mel_db = 20.0 * torch.log10(torch.clamp(mel, min=1e-5)) - 20.0
        min_level_db = -100.0
        max_norm = 4.0
        mel_norm = ((mel_db - min_level_db) / (-min_level_db)) * 2.0 * max_norm - max_norm
        mel_norm = torch.clamp(mel_norm, -max_norm, max_norm)
        return mel_norm


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
        if pw is not None:
            wav_np = flat.numpy().astype(np.float64)
            f0_ts, time_axis = pw.dio(wav_np, sr, frame_period=5.0)
            f0_np = pw.stonemask(wav_np, f0_ts, time_axis, sr)
            f0 = torch.from_numpy(np.ascontiguousarray(f0_np)).float()
        else:
            pitch = AF.detect_pitch_frequency(flat, sample_rate=sr).flatten().float()
            f0 = pitch
        f0 = _interp_to(f0, num_frames)
    except Exception:
        return zeros(), zeros()

    voiced_mask = (f0 > 0.0).float()
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
