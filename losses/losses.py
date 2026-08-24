"""
Loss Functions for TamilTTSv2 (FastPitch Variant + FiLM Style + Staged GAN)
===========================================================================
- masked_l1: generic length-masked L1 over dim=1 with trailing-dim support.
- MelLoss: dual masked L1 for refined (mel_pred) and coarse mel spectrograms.
- DurationLoss: masked L1 in log-domain against alignment-derived durations.
- PitchEnergyLoss: voiced-restricted log-F0 L1 + masked energy L1.
- DiscriminatorLoss: LSGAN hinge-free least-squares discriminator objective.
- GeneratorAdversarialLoss: LSGAN generator (non-saturating LS) objective.
- FeatureMatchingLoss: layer-wise L1 feature matching averaged across discriminators.
- SLMLoss: multi-layer WavLM perceptual feature-matching loss (Stage 3).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


def masked_l1(pred, target, lens):
    """
    Mean L1 loss over valid positions along dim=1, supporting extra trailing dims.

    Args:
        pred:   [B, T] or [B, T, ...]
        target: same shape as pred
        lens:   [B] valid lengths along dim=1, or None for unmasked mean
    """
    min_t = min(pred.size(1), target.size(1))
    pred = pred[:, :min_t]
    target = target[:, :min_t]
    loss = F.l1_loss(pred.float(), target.float(), reduction="none")
    if lens is None:
        return loss.mean()
    lens = lens.to(loss.device).long()
    mask = (torch.arange(min_t, device=loss.device).unsqueeze(0) < lens.unsqueeze(1)).float()
    broadcast_shape = (mask.size(0), min_t) + (1,) * (loss.dim() - 2)
    mask = mask.view(broadcast_shape)
    per_position = loss.numel() // max(loss.size(0) * loss.size(1), 1)
    denom = torch.clamp(mask.sum() * per_position, min=1.0)
    return (loss * mask).sum() / denom


class SpectralConvergenceLoss(nn.Module):
    """
    Spectral Convergence Loss: ||S_true - S_pred||_F / ||S_true||_F.
    Forces sharp spectral peakiness and penalizes blurry conditional-mean mel predictions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, mel_pred, mel_target, mel_lens=None):
        min_t = min(mel_pred.size(1), mel_target.size(1))
        pred = mel_pred[:, :min_t]
        target = mel_target[:, :min_t]

        if mel_lens is not None:
            lens = mel_lens.to(pred.device).long()
            mask = (torch.arange(min_t, device=pred.device).unsqueeze(0) < lens.unsqueeze(1)).float()
            pred = pred * mask.unsqueeze(-1)
            target = target * mask.unsqueeze(-1)

        diff = (target.float() - pred.float()).norm(p="fro", dim=(-2, -1))
        denom = torch.clamp(target.float().norm(p="fro", dim=(-2, -1)), min=1e-5)
        return (diff / denom).mean()


class MelLoss(nn.Module):
    """
    Masked Dual Mel-Spectrogram Loss + Spectral Convergence.
    Targets are the normalized [B, 80, Tm] mel transposed to [B, Tm, 80].
    `lowband_w` multiplies the L1 on the lowest `lowband_bins` mel bins: measured
    +0.46 low-frequency bias (drone) in trained models needs extra pressure there.
    Returns (weighted_total, l_refined, l_coarse).
    """

    def __init__(self, coarse_w=0.5, refined_w=1.0, sc_w=1.0, lowband_w=1.0, lowband_bins=10):
        super().__init__()
        self.coarse_w = coarse_w
        self.refined_w = refined_w
        self.sc_w = sc_w
        self.lowband_w = lowband_w
        self.lowband_bins = lowband_bins
        self.sc_loss = SpectralConvergenceLoss()

    def _masked_l1_binned(self, pred, target, lens):
        """Per-bin masked L1: [B, Tm, 80] -> [80]."""
        min_t = min(pred.size(1), target.size(1))
        pred = pred[:, :min_t]
        target = target[:, :min_t]
        loss = F.l1_loss(pred.float(), target.float(), reduction="none")  # [B,Tm,80]
        if lens is None:
            return loss.mean(dim=(0, 1))
        lens = lens.to(loss.device).long()
        mask = (torch.arange(min_t, device=loss.device).unsqueeze(0) < lens.unsqueeze(1)).float()
        denom = torch.clamp(mask.sum(), min=1.0)
        return (loss * mask.unsqueeze(-1)).sum(dim=(0, 1)) / denom

    def forward(self, mel_pred, mel_coarse, mel_target, mel_lens=None):
        target = mel_target.transpose(1, 2)
        # per-bin losses enable targeted low-band weighting
        bins_ref = self._masked_l1_binned(mel_pred, target, mel_lens)
        bins_crs = self._masked_l1_binned(mel_coarse, target, mel_lens)
        w = torch.ones_like(bins_ref)
        if self.lowband_w != 1.0:
            w[:self.lowband_bins] = self.lowband_w
        l_refined = (bins_ref * w).mean()
        l_coarse = (bins_crs * w).mean()
        l_sc = self.sc_loss(mel_pred, target, mel_lens) if self.sc_w > 0 else torch.zeros((), device=mel_pred.device)
        total = self.refined_w * l_refined + self.coarse_w * l_coarse + self.sc_w * l_sc
        return total, l_refined, l_coarse


class DurationLoss(nn.Module):
    """Masked L1 between predicted log-durations and log of clamped GT durations."""

    def __init__(self):
        super().__init__()

    def forward(self, log_dur_pred, gt_dur, token_lens=None):
        target_log_dur = torch.log(gt_dur.float().clamp(min=1.0))
        return masked_l1(log_dur_pred, target_log_dur, token_lens)


class PitchEnergyLoss(nn.Module):
    """
    Masked prosody losses.
    - log_f0 supervised only on frames where (voiced == 1) AND frame < mel_len.
    - energy supervised on all valid frames.
    Returns (f0_loss, energy_loss).
    """

    def forward(self, f0_pred, energy_pred, f0_target, voiced, energy_target, mel_lens=None):
        def _sq(x):
            if x.dim() == 3 and x.size(-1) == 1:
                return x.squeeze(-1)
            return x

        f0_pred = _sq(f0_pred).float()
        f0_target = _sq(f0_target).float()
        energy_pred = _sq(energy_pred).float()
        energy_target = _sq(energy_target).float()
        voiced = _sq(voiced).float()

        min_t = min(
            f0_pred.size(1), f0_target.size(1),
            energy_pred.size(1), energy_target.size(1), voiced.size(1),
        )
        f0_pred, f0_target = f0_pred[:, :min_t], f0_target[:, :min_t]
        energy_pred, energy_target = energy_pred[:, :min_t], energy_target[:, :min_t]
        voiced = voiced[:, :min_t]

        idx = torch.arange(min_t, device=f0_pred.device).unsqueeze(0)
        if mel_lens is not None:
            valid = idx < mel_lens.to(f0_pred.device).unsqueeze(1)
        else:
            valid = torch.ones_like(idx, dtype=torch.bool).expand_as(voiced.bool())

        voiced_mask = (valid & (voiced > 0.5)).float()
        valid_mask = valid.float()

        f0_loss = ((f0_pred - f0_target).abs() * voiced_mask).sum() / torch.clamp(voiced_mask.sum(), min=1.0)
        energy_loss = ((energy_pred - energy_target).abs() * valid_mask).sum() / torch.clamp(valid_mask.sum(), min=1.0)
        return f0_loss, energy_loss


class DiscriminatorLoss(nn.Module):
    """
    LSGAN discriminator objective (Mao et al.).
    d_loss = mean_over_scores(real: (s-1)^2) + mean_over_scores(fake: s^2).
    """

    def forward(self, scores_real, scores_fake):
        if not len(scores_real) and not len(scores_fake):
            return torch.zeros((), device="cpu")
        d_real = torch.stack([((s.float() - 1.0) ** 2).mean() for s in scores_real]).mean()
        d_fake = torch.stack([(s.float() ** 2).mean() for s in scores_fake]).mean()
        return d_real + d_fake


class GeneratorAdversarialLoss(nn.Module):
    """LSGAN generator objective: sum over fake score tensors of (s-1)^2."""

    def forward(self, scores_fake):
        if not len(scores_fake):
            return torch.zeros((), device="cpu")
        return sum(((s.float() - 1.0) ** 2).mean() for s in scores_fake)


class FeatureMatchingLoss(nn.Module):
    """
    Layer-wise L1 between fake and real discriminator features,
    averaged within each discriminator then across discriminators.
    Accepts arbitrarily nested per-discriminator layer lists.
    """

    @staticmethod
    def _iter_layers(entry):
        if isinstance(entry, (list, tuple)):
            for item in entry:
                yield from FeatureMatchingLoss._iter_layers(item)
        else:
            yield entry

    def forward(self, feats_fake, feats_real):
        disc_losses = []
        for fake_group, real_group in zip(feats_fake, feats_real):
            layer_losses = []
            for fk, rl in zip(self._iter_layers(fake_group), self._iter_layers(real_group)):
                layer_losses.append(F.l1_loss(fk.float(), rl.detach().float()))
            if layer_losses:
                disc_losses.append(torch.stack(layer_losses).mean())
        if not disc_losses:
            return torch.zeros((), device="cpu")
        return torch.stack(disc_losses).mean()


class SLMLoss(nn.Module):
    """
    Speech Language Model (SLM) Multi-Layer WavLM Feature Matching Loss.
    Resamples waveforms to 16 kHz, compares WavLM hidden_states layers [3, 7, 11]
    with L1; falls back to last_hidden_state when layers are unavailable.
    """

    def __init__(self, wavlm_model, sample_rate=22050, target_sr=16000):
        super().__init__()
        self.wavlm = wavlm_model
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.wavlm.eval()
        self.target_layers = [3, 7, 11]

        self.resampler = None
        if sample_rate != target_sr:
            self.resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)

    def forward(self, real_audio, gen_audio):
        real_audio = real_audio.float()
        gen_audio = gen_audio.float()

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
        gen_out = self.wavlm(gen_audio, output_hidden_states=True)

        real_states = real_out.hidden_states
        gen_states = gen_out.hidden_states

        layer_losses = []
        for l_idx in self.target_layers:
            if l_idx < len(real_states):
                r_feat = real_states[l_idx].detach()
                g_feat = gen_states[l_idx]
                min_t = min(r_feat.size(1), g_feat.size(1))
                layer_losses.append(F.l1_loss(g_feat[:, :min_t, :], r_feat[:, :min_t, :]))

        if layer_losses:
            return sum(layer_losses) / len(layer_losses)

        min_t = min(real_out.last_hidden_state.size(1), gen_out.last_hidden_state.size(1))
        return F.l1_loss(
            gen_out.last_hidden_state[:, :min_t, :],
            real_out.last_hidden_state[:, :min_t, :].detach(),
        )
