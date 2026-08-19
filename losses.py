import torch
import torch.nn as nn
import torch.nn.functional as F

class SLMLoss(nn.Module):
    """WavLM based Adversarial Critic"""
    def __init__(self, wavlm_model):
        super().__init__()
        self.wavlm = wavlm_model
        
    def forward(self, real_audio, generated_audio):
        real_features = self.wavlm(real_audio).last_hidden_state
        gen_features = self.wavlm(generated_audio).last_hidden_state
        # Simplified Hinge Loss
        loss = F.relu(1.0 - real_features.mean()) + F.relu(1.0 + gen_features.mean())
        return loss

class SRFDLoss(nn.Module):
    """Speech Representation Frechet Distance using AI4Bharat IndicWhisper"""
    def __init__(self, whisper_encoder):
        super().__init__()
        self.whisper = whisper_encoder
        
    def forward(self, real_audio, generated_audio, feature_extractor):
        # Whisper requires specific input feature extraction
        real_mel = feature_extractor(real_audio.cpu().numpy(), sampling_rate=16000, return_tensors='pt').input_features.to(self.whisper.device, dtype=self.whisper.dtype)
        gen_mel = feature_extractor(generated_audio.cpu().numpy(), sampling_rate=16000, return_tensors='pt').input_features.to(self.whisper.device, dtype=self.whisper.dtype)
        
        real_features = self.whisper(real_mel).last_hidden_state
        gen_features = self.whisper(gen_mel).last_hidden_state
        
        # Calculate Frechet Distance (Simplified to MSE for skeleton)
        loss = F.mse_loss(real_features.mean(dim=1), gen_features.mean(dim=1))
        return loss
