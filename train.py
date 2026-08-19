import torch
import torch.nn as nn
from models import TamilTTS
from losses import SLMLoss, SRFDLoss
from dataset import ShrutilipiDataset
from lion_pytorch import Lion

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize Core Model
    model = TamilTTS().to(device)
    
    # --- MULTI-GPU DETECTION ---
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"🔥 Detected {gpu_count} GPUs! Enabling Multi-GPU DataParallel Training...")
        model = nn.DataParallel(model)
    else:
        print(f"Detected {gpu_count} GPU. Single-GPU Training.")
        
    optimizer = Lion(model.parameters(), lr=1e-4)
    
    print("Model initialized on:", device)
    
    # 2. Load Critics (Paths will be passed in via notebook)
    print("Ready to begin training loop with SR-FD and SLM Losses!")

if __name__ == '__main__':
    train()
