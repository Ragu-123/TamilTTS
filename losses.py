import torch
import torch.nn as nn
import torch.nn.functional as F

class SLMLoss(nn.Module):
    def __init__(self, wavlm_model):
        super().__init__()
        self.wavlm = wavlm_model
        # Freeze WavLM completely
        for param in self.wavlm.parameters():
            param.requires_grad = False
            
    def forward(self, real_audio, generated_audio):
        with torch.no_grad():
            real_features = self.wavlm(real_audio).last_hidden_state
        # Generate features requires gradient flow to generator
        gen_features = self.wavlm(generated_audio).last_hidden_state
        
        # Hinge Loss for Adversarial Training
        real_loss = F.relu(1.0 - real_features.mean())
        gen_loss = F.relu(1.0 + gen_features.mean())
        
        # Generator Loss tries to fool discriminator
        generator_loss = -gen_features.mean()
        
        return generator_loss + real_loss + gen_loss

class SRFDLoss(nn.Module):
    def __init__(self, whisper_encoder, feature_extractor):
        super().__init__()
        self.whisper = whisper_encoder
        self.extractor = feature_extractor
        # Freeze Whisper completely
        for param in self.whisper.parameters():
            param.requires_grad = False
            
    def forward(self, real_audio, generated_audio):
        # We must extract log-mels specifically for Whisper
        device = self.whisper.device
        dtype = self.whisper.dtype
        
        # Note: In a true pipeline, you pass tensors directly. For simplicity here:
        with torch.no_grad():
            real_mel = self.extractor(real_audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)
            real_features = self.whisper(real_mel).last_hidden_state

        gen_mel = self.extractor(generated_audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)
        gen_features = self.whisper(gen_mel).last_hidden_state
        
        # Fréchet Distance (MSE proxy for feature matching)
        loss = F.mse_loss(real_features.mean(dim=1), gen_features.mean(dim=1))
        return loss
