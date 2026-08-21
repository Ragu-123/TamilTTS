import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class AudioMelLoss(nn.Module):
    """
    HiFi-GAN Standard Differentiable Mel-Spectrogram Loss.
    Computes mel spectrograms directly from generated audio waveform
    and computes L1 loss against the ground-truth audio mel spectrogram.
    
    This is the critical loss that directly trains the FullVocoder to synthesize
    exact acoustic frequencies and human vocal tract resonances.
    """
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
            normalized=False,
        )

    def forward(self, gen_audio, real_audio):
        device = gen_audio.device
        mel_tf = self.mel_transform.to(device)

        gen_mel = torch.log(torch.clamp(mel_tf(gen_audio), min=1e-5))
        with torch.no_grad():
            real_mel = torch.log(torch.clamp(mel_tf(real_audio), min=1e-5))

        min_t = min(gen_mel.size(-1), real_mel.size(-1))
        return F.l1_loss(gen_mel[..., :min_t], real_mel[..., :min_t])


class SLMLoss(nn.Module):
    """WavLM adversarial feature-matching loss. Input: [B, T] raw waveform."""
    def __init__(self, wavlm_model):
        super().__init__()
        self.wavlm = wavlm_model
        for p in self.wavlm.parameters():
            p.requires_grad = False
        self.wavlm.eval()

    def forward(self, real_audio, gen_audio):
        with torch.no_grad():
            real_feat = self.wavlm(real_audio).last_hidden_state
        gen_feat = self.wavlm(gen_audio).last_hidden_state
        min_t = min(real_feat.size(1), gen_feat.size(1))
        return F.l1_loss(gen_feat[:, :min_t, :], real_feat[:, :min_t, :].detach())


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
