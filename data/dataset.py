import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class ShrutilipiDataset(Dataset):
    """
    Loads Shrutilipi Tamil parquet files.
    Columns: audio_filepath, text, duration, lang
    """
    def __init__(self, parquet_path, max_audio_len=48000, max_text_len=200):
        print("Loading dataset (this may take a few minutes)...")
        self.ds = load_dataset("parquet", data_dir=parquet_path, split="train")
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

        # Build character vocab from Tamil Unicode block
        self.char2id = {" ": 1}  # 0 = PAD, 1 = SPACE
        idx = 2
        for c in range(0x0B80, 0x0C00):  # Tamil Unicode block
            self.char2id[chr(c)] = idx
            idx += 1
        # Common punctuation
        for p in ".,!?;:-\'":
            self.char2id[p] = idx
            idx += 1

    def __len__(self):
        return len(self.ds)

    def text_to_ids(self, text):
        ids = []
        for ch in text:
            ids.append(self.char2id.get(ch, 0))
        # Pad / truncate
        ids = ids[:self.max_text_len]
        ids = ids + [0] * (self.max_text_len - len(ids))
        return ids

    def __getitem__(self, idx):
        sample = self.ds[idx]

        # --- Text ---
        text = sample["text"]
        token_ids = torch.tensor(self.text_to_ids(text), dtype=torch.long)

        # --- Audio ---
        audio_data = sample["audio_filepath"]
        audio_array = np.array(audio_data["array"], dtype=np.float32)
        sr = audio_data["sampling_rate"]

        if sr != 16000:
            audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=16000)

        # Pad or truncate
        if len(audio_array) > self.max_audio_len:
            audio_array = audio_array[:self.max_audio_len]
        else:
            audio_array = np.pad(audio_array, (0, self.max_audio_len - len(audio_array)))

        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

        # --- Mel Spectrogram (80-band) for Style Encoder ---
        mel = librosa.feature.melspectrogram(
            y=audio_array, sr=16000, n_fft=1024,
            hop_length=256, n_mels=80
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32)  # [80, T_mel]

        return token_ids, mel_tensor, audio_tensor


def get_dataloader(cfg):
    ds = ShrutilipiDataset(
        cfg.dataset_dir,
        max_audio_len=cfg.max_audio_len,
        max_text_len=cfg.max_text_len,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
