import torch
import torch.nn as nn
import torch.nn.functional as F

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
