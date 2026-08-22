"""
Loss Functions for TamilTTS (FastPitch / RAD-TTS / StyleTTS2 SOTA Standard)
===========================================================================
- DualMelLoss: Masked L1 Loss for Coarse & PostNet Refined Mel Spectrograms.
- LogDurationLoss: Masked Duration Loss between Predicted and Alignment-Derived Durations.
- SLMLoss: Multi-layer WavLM Perceptual Feature-matching Loss (Stage 2/3).
- SRFDLoss: IndicWhisper Speech Representation Fréchet Distance (Validation Metric).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class DualMelLoss(nn.Module):
    """
    Masked Dual Mel-Spectrogram Loss (FastPitch / Tacotron 2 standard).
    Computes L1 loss strictly over valid acoustic frames, completely ignoring padding frames.
    """
    def __init__(self, coarse_weight=0.5, refined_weight=1.0):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.refined_weight = refined_weight

    def forward(self, mel_refined, mel_coarse, mel_target, mel_lens=None):
        """
        mel_refined: [B, T_mel, 80]
        mel_coarse:  [B, T_mel, 80]
        mel_target:  [B, T_mel, 80]
        mel_lens:    [B] actual valid frame counts per sample
        """
        min_t = min(mel_refined.size(1), mel_target.size(1), mel_coarse.size(1))
        mel_refined = mel_refined[:, :min_t]
        mel_coarse  = mel_coarse[:, :min_t]
        mel_target  = mel_target[:, :min_t]

        if mel_lens is not None:
            # Create boolean mask: [B, min_t, 1] (True for valid speech frames)
            device = mel_refined.device
            mask = (torch.arange(min_t, device=device).unsqueeze(0) < mel_lens.unsqueeze(1)).unsqueeze(-1).float()
            denom = mask.sum() * 80.0 + 1e-6

            loss_refined = (F.l1_loss(mel_refined, mel_target, reduction='none') * mask).sum() / denom
            loss_coarse  = (F.l1_loss(mel_coarse, mel_target, reduction='none') * mask).sum() / denom
        else:
            loss_refined = F.l1_loss(mel_refined, mel_target)
            loss_coarse  = F.l1_loss(mel_coarse, mel_target)

        total_mel_loss = self.refined_weight * loss_refined + self.coarse_weight * loss_coarse
        return total_mel_loss, loss_refined, loss_coarse


class LogDurationLoss(nn.Module):
    """
    Masked Log-Scale Duration Loss.
    Supervises the DurationPredictor against exact alignment-derived durations.
    """
    def __init__(self):
        super().__init__()

    def forward(self, log_dur_pred, target_dur, text_lens=None):
        """
        log_dur_pred: [B, T_text] (predicted log-durations)
        target_dur:   [B, T_text] (alignment-derived true frame counts)
        text_lens:    [B] (actual character lengths)
        """
        target_log_dur = torch.log(target_dur.float().clamp(min=1e-5))
        loss = F.mse_loss(log_dur_pred, target_log_dur, reduction='none')

        if text_lens is not None:
            device = log_dur_pred.device
            mask = (torch.arange(log_dur_pred.size(1), device=device).unsqueeze(0) < text_lens.unsqueeze(1)).float()
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss


class PitchLoss(nn.Module):
    """
    Masked Log-F0 Pitch Loss.
    Supervises the PitchPredictor against continuous pitch contours.
    """
    def __init__(self):
        super().__init__()

    def forward(self, f0_pred, f0_target, mel_lens=None):
        """
        f0_pred:   [B, T_mel, 1]
        f0_target: [B, T_mel, 1]
        mel_lens:  [B]
        """
        min_t = min(f0_pred.size(1), f0_target.size(1))
        f0_pred = f0_pred[:, :min_t]
        f0_target = f0_target[:, :min_t]

        loss = F.mse_loss(f0_pred, f0_target, reduction='none')

        if mel_lens is not None:
            device = f0_pred.device
            mask = (torch.arange(min_t, device=device).unsqueeze(0) < mel_lens.unsqueeze(1)).unsqueeze(-1).float()
            loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss


class SLMLoss(nn.Module):
    """
    Speech Language Model (SLM) Multi-Layer WavLM Feature Matching Loss.
    Resamples 22.05 kHz generated/real waveforms to 16.0 kHz for WavLM evaluation.
    """
    def __init__(self, wavlm_model, sample_rate=22050, target_sr=16000):
        super().__init__()
        self.wavlm = wavlm_model
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.wavlm.eval()

        self.resampler = None
        if sample_rate != target_sr:
            self.resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)

    def forward(self, real_audio, gen_audio):
        if self.resampler is not None:
            device = gen_audio.device
            resamp = self.resampler.to(device)
            real_audio = resamp(real_audio)
            gen_audio = resamp(gen_audio)

        min_len = min(real_audio.size(-1), gen_audio.size(-1))
        real_audio = real_audio[..., :min_len]
        gen_audio = gen_audio[..., :min_len]

        with torch.no_grad():
            real_out = self.wavlm(real_audio, output_hidden_states=True)
            real_states = real_out.hidden_states

        gen_out = self.wavlm(gen_audio, output_hidden_states=True)
        gen_states = gen_out.hidden_states

        target_layers = [3, 6, 9, 12]
        layer_losses = []
        for l_idx in target_layers:
            if l_idx < len(real_states):
                r_feat = real_states[l_idx].detach()
                g_feat = gen_states[l_idx]
                min_t = min(r_feat.size(1), g_feat.size(1))
                layer_losses.append(F.l1_loss(g_feat[:, :min_t, :], r_feat[:, :min_t, :]))

        if layer_losses:
            return sum(layer_losses) / len(layer_losses)
        else:
            min_t = min(real_out.last_hidden_state.size(1), gen_out.last_hidden_state.size(1))
            return F.l1_loss(gen_out.last_hidden_state[:, :min_t, :], real_out.last_hidden_state[:, :min_t, :].detach())


class SRFDLoss(nn.Module):
    """IndicWhisper Speech Representation Fréchet Distance (Validation only)."""
    def __init__(self, whisper_encoder, feature_extractor, sample_rate=22050):
        super().__init__()
        self.whisper = whisper_encoder
        self.extractor = feature_extractor
        for p in self.whisper.parameters():
            p.requires_grad = False
        self.whisper.eval()
        self.resampler = T.Resample(orig_freq=sample_rate, new_freq=16000) if sample_rate != 16000 else None

    @torch.no_grad()
    def forward(self, real_audio, gen_audio):
        device = next(self.whisper.parameters()).device
        dtype  = next(self.whisper.parameters()).dtype

        if self.resampler is not None:
            resamp = self.resampler.to(device)
            real_audio = resamp(real_audio)
            gen_audio = resamp(gen_audio)

        real_mel = self.extractor(
            real_audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=dtype)
        gen_mel = self.extractor(
            gen_audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=dtype)
        real_feat = self.whisper(real_mel).last_hidden_state
        gen_feat  = self.whisper(gen_mel).last_hidden_state
        return F.mse_loss(gen_feat.float().mean(dim=1), real_feat.float().mean(dim=1))
