import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import TamilTTS
from losses import SLMLoss, SRFDLoss
from dataset import get_dataloader
from lion_pytorch import Lion
from transformers import WhisperModel, WhisperFeatureExtractor, WavLMModel

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Initialize Generator (TamilTTS Architecture)
    model = TamilTTS().to(device)
    
    # Enable Multi-GPU
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"🔥 Detected {gpu_count} GPUs! Enabling Multi-GPU DataParallel Training...")
        model = nn.DataParallel(model)
    
    optimizer = Lion(model.parameters(), lr=1e-4)
    print("Model and Optimizer Initialized!")

    # 2. Initialize Critics (WavLM and IndicWhisper)
    print(f"Loading SLM Critic (WavLM) from {args.wavlm_dir}...")
    wavlm_model = WavLMModel.from_pretrained(args.wavlm_dir).to(device).eval()
    slm_criterion = SLMLoss(wavlm_model)
    
    print(f"Loading SR-FD Critic (IndicWhisper) from {args.whisper_dir}...")
    whisper_encoder = WhisperModel.from_pretrained(args.whisper_dir).encoder.to(device).eval()
    whisper_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper_dir)
    srfd_criterion = SRFDLoss(whisper_encoder, whisper_extractor)

    # 3. Load Dataset
    print(f"Loading Shrutilipi Dataset from {args.dataset_dir}...")
    dataloader = get_dataloader(args.dataset_dir, batch_size=8)
    
    # 4. Main Training Loop
    print("Starting Training Loop...")
    for epoch in range(100):
        for step, (text_tokens, ref_mel, real_audio) in enumerate(dataloader):
            text_tokens = text_tokens.to(device)
            ref_mel = ref_mel.to(device)
            real_audio = real_audio.to(device)
            
            diffusion_time = torch.rand(real_audio.size(0)).to(device)
            
            # Forward Pass: Generate Audio
            optimizer.zero_grad()
            generated_audio, prosody = model(text_tokens, ref_mel, diffusion_time)
            
            # Calculate Advanced Losses
            loss_slm = slm_criterion(real_audio, generated_audio)
            loss_srfd = srfd_criterion(real_audio, generated_audio)
            
            # Total Loss Combination
            total_loss = loss_slm + loss_srfd
            
            # Backpropagation
            total_loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Epoch: {epoch} | Step: {step} | Total Loss: {total_loss.item():.4f} | SLM Loss: {loss_slm.item():.4f} | SR-FD Loss: {loss_srfd.item():.4f}")
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help="Path to Shrutilipi Parquet")
    parser.add_argument('--wavlm_dir', type=str, required=True, help="Path to WavLM model")
    parser.add_argument('--whisper_dir', type=str, required=True, help="Path to IndicWhisper model")
    args = parser.parse_args()
    train(args)
