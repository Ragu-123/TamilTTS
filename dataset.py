import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import librosa
from indicnlp.tokenize import indic_tokenize

class ShrutilipiDataset(Dataset):
    def __init__(self, parquet_path):
        self.ds = load_dataset('parquet', data_dir=parquet_path, split='train')
        self.vocab = {chr(i): i-2944 for i in range(2944, 3072)} # Tamil Unicode block mapping approximation
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        sample = self.ds[idx]
        text = sample['transcript']
        
        tokens = indic_tokenize.trivial_tokenize(text, lang='ta')
        token_ids = []
        for word in tokens:
            for char in word:
                if char in self.vocab:
                    token_ids.append(self.vocab[char])
                else:
                    token_ids.append(0) # UNK
                    
        # Pad or truncate tokens to 200 length
        token_ids = token_ids[:200] + [0] * (200 - len(token_ids))
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        
        audio_array = sample['audio_filepath']['array']
        sr = sample['audio_filepath']['sampling_rate']
        
        if sr != 16000:
            audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=16000)
            
        # Ensure fixed audio length of 30,000 samples (roughly 1.8 seconds) for batching
        target_len = 30000
        if len(audio_array) > target_len:
            audio_array = audio_array[:target_len]
        else:
            audio_array = torch.nn.functional.pad(torch.tensor(audio_array), (0, target_len - len(audio_array))).numpy()
            
        mel = torch.randn(80, 200) # Dummy Mel placeholder for Style Encoder
        
        return token_tensor, mel, torch.tensor(audio_array).float()

def get_dataloader(dataset_path, batch_size=4):
    ds = ShrutilipiDataset(dataset_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
