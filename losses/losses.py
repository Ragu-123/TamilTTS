"""
Loss Functions for TamilTTS (StyleTTS 2 / Kokoro-82M / FastPitch SOTA Standard)
==============================================================================
- DualMelLoss: Coarse Mel Loss (Pre-PostNet) + Refined Mel Loss (Post-PostNet).
- LogDurationLoss: Masked Log-scale Duration Loss.
- SLMLoss: StyleTTS 2 Multi-layer WavLM Perceptual Feature-matching Loss.
- SRFDLoss: IndicWhisper Speech Representation Fréchet Distance (Validation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class DualMelLoss(nn.Module):
    """
    Dual Mel-Spectrogram Loss (Tacotron 2 / Kokoro-82M standard).
    Supervises both the coarse acoustic projection and the 5-layer PostNet refinement.
    """
    def __init__(self, coarse_weight=0.5, refined_weight=1.0):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.refined_weight = refined_weight

    def forward(self, mel_refined, mel_coarse, mel_target):
        """
        mel_refined: [B, T_mel, 80]
        mel_coarse:  [B, T_mel, 80]
        mel_target:  [B, T_mel, 80]
        """
        min_t = min(mel_refined.size(1), mel_target.size(1), mel_coarse.size(1))
        loss_refined = F.l1_loss(mel_refined[:, :min_t], mel_target[:, :min_t])
        loss_coarse = F.l1_loss(mel_coarse[:, :min_t], mel_target[:, :min_t])
        total_mel_loss = self.refined_weight * loss_refined + self.coarse_weight * loss_coarse
        return total_mel_loss, loss_refined, loss_coarse


class LogDurationLoss(nn.Module):
    """
    Masked Log-Scale Duration Loss (Kokoro-82M / FastPitch standard).
    Penalizes relative errors proportionally across consonants and vowels.
    """
    def __init__(self):
        super().__init__()

    def forward(self, log_dur_pred, target_dur, mask=None):
        """
        log_dur_pred: [B, T_text] (predicted log-durations)
        target_dur:   [B, T_text] (ground-truth MAS frame counts)
        mask:         [B, T_text] (True for PAD)
        """
        target_log_dur = torch.log(target_dur.float().clamp(min=1e-5))
        loss = F.mse_loss(log_dur_pred, target_log_dur, reduction='none')

        if mask is not None:
            non_pad = (~mask).float()
            loss = (loss * non_pad).sum() / (non_pad.sum() + 1e-6)
        else:
            loss = loss.mean()
        return loss


class SLMLoss(nn.Module):
    """
    StyleTTS 2 / Kokoro-82M Speech Language Model (SLM) Multi-Layer Feature Matching Loss.
    
    Extracts intermediate representations from WavLM layers [3, 6, 9, 12] to evaluate:
    - Layers 3 & 6: Phonetic precision, formant sharpness, and consonant articulation.
    - Layers 9 & 12: Speaker timbre, prosody, and natural human vocal tone.
    """
    def __init__(self, wavlm_model, sample_rate=16000, target_sr=16000):
        super().__init__()
        self.wavlm = wavlm_model
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.wavlm.eval()

        self.resampler = None
        if sample_rate != target_sr:
            self.resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)

    def forward(self, real_audio, gen_audio):
        """
        real_audio: [B, T]
        gen_audio:  [B, T]
        """
        if self.resampler is not None:
            device = gen_audio.device
            resamp = self.resampler.to(device)
            real_audio = resamp(real_audio)
            gen_audio = resamp(gen_audio)

        min_len = min(real_audio.size(-1), gen_audio.size(-1))
        real_audio = real_audio[..., :min_len]
        gen_audio = gen_audio[..., :min_len]

        # Extract multi-layer hidden states from WavLM
        with torch.no_grad():
            real_out = self.wavlm(real_audio, output_hidden_states=True)
            real_states = real_out.hidden_states  # Tuple of 13 layer tensors

        gen_out = self.wavlm(gen_audio, output_hidden_states=True)
        gen_states = gen_out.hidden_states

        # Compare across layers [3, 6, 9, 12] (StyleTTS 2 standard)
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
    """IndicWhisper SR-FD metric (validation only, no gradients)."""
    def __init__(self, whisper_encoder, feature_extractor):
        super().__init__()
        self.whisper = whisper_encoder
        self.extractor = feature_extractor
        for p in self.whisper.parameters():
            p.requires_grad = False
        self.whisper.eval()

    @torch.no_grad()
    def forward(self, real_audio, gen_audio):
        device = next(self.whisper.parameters()).device
        dtype  = next(self.whisper.parameters()).dtype
        real_mel = self.extractor(
            real_audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=dtype)
        gen_mel = self.extractor(
            gen_audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=dtype)
        real_feat = self.whisper(real_mel).last_hidden_state
        gen_feat  = self.whisper(gen_mel).last_hidden_state
        return F.mse_loss(gen_feat.float().mean(dim=1), real_feat.float().mean(dim=1))
