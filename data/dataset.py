import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

class ShrutilipiDataset(Dataset):
    """
    Shrutilipi Tamil parquet dataset.
    Columns: audio_filepath (dict with array+sr), text, duration, lang
    """
    def __init__(self, hf_dataset, max_audio_len=48000, max_text_len=200,
                 mel_channels=80, n_fft=1024, hop_length=256, sr=16000):
        self.ds = hf_dataset
        self.max_audio_len = max_audio_len
        self.max_text_len  = max_text_len
        self.mel_channels  = mel_channels
        self.n_fft         = n_fft
        self.hop_length    = hop_length
        self.sr            = sr

        # Tamil Unicode character vocabulary
        self.char2id = {" ": 1}  # 0=PAD, 1=SPACE
        idx = 2
        for c in range(0x0B80, 0x0C00):
            self.char2id[chr(c)] = idx
            idx += 1
        for p in list(".,!?;:-"):
            self.char2id[p] = idx
            idx += 1

    def __len__(self):
        return len(self.ds)

    def text_to_ids(self, text):
        ids = [self.char2id.get(ch, 0) for ch in text]
        ids = ids[:self.max_text_len]
        ids += [0] * (self.max_text_len - len(ids))
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

        if sr != self.sr:
            audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=self.sr)

        # Pad or truncate to fixed length
        if len(audio_array) > self.max_audio_len:
            audio_array = audio_array[:self.max_audio_len]
        else:
            audio_array = np.pad(audio_array, (0, self.max_audio_len - len(audio_array)))

        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

        # --- Mel Spectrogram (80-band) for Style Encoder ---
        mel = librosa.feature.melspectrogram(
            y=audio_array, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.mel_channels,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.tensor(mel_db, dtype=torch.float32)  # [80, T_mel]

        return token_ids, mel_tensor, audio_tensor


def build_dataloaders(cfg):
    """Load dataset and split into train (95%) and validation (5%)."""
    print("Loading Shrutilipi dataset...")
    full_ds = load_dataset("parquet", data_dir=cfg.dataset_dir, split="train")

    # Split
    split = full_ds.train_test_split(test_size=cfg.val_split, seed=42)
    train_hf = split["train"]
    val_hf   = split["test"]

    print(f"  Train samples: {len(train_hf)}")
    print(f"  Val samples  : {len(val_hf)}")

    train_ds = ShrutilipiDataset(
        train_hf, max_audio_len=cfg.max_audio_len, max_text_len=cfg.max_text_len,
        mel_channels=cfg.mel_channels, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )
    val_ds = ShrutilipiDataset(
        val_hf, max_audio_len=cfg.max_audio_len, max_text_len=cfg.max_text_len,
        mel_channels=cfg.mel_channels, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader
