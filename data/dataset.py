import os
import re
import glob
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# Initialize Tamil normalizer once (module-level, shared across all workers)
_tamil_normalizer = IndicNormalizerFactory().get_normalizer("ta")

# Regex to normalize subscript digits (₀-₉) to standard digits (0-9)
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_tamil_text(text):
    """
    Clean Tamil text using indic-nlp-library + custom rules.
    This runs BEFORE character tokenization.

    What it does:
    1. indic-nlp normalizer: fixes Unicode variations of Tamil chars,
       removes zero-width joiners/non-joiners, standardizes punctuation
    2. Subscript digits: ₀₁₂₃ -> 0123
    3. Strip extra whitespace
    """
    if not isinstance(text, str):
        return ""
    text = _tamil_normalizer.normalize(text)
    text = text.translate(_SUBSCRIPT_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TamilTTSDataset(Dataset):
    """
    Universal High-Fidelity Tamil TTS Dataset.
    Supports:
      - AI4Bharat Rasa (Studio Expressive TTS): ['text', 'audio', 'gender', 'style', 'duration']
      - AI4Bharat IndicVoices-R (Clean Read Speech): ['normalized', 'text', 'audio', 'verbatim']
      - AI4Bharat Shrutilipi: ['text', 'audio_filepath']
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
        self.max_mel_len   = max_audio_len // hop_length  # 188

        # Tamil Unicode character vocabulary (0x0B80-0x0BFF)
        self.char2id = {" ": 1}  # 0=PAD, 1=SPACE
        idx = 2
        for c in range(0x0B80, 0x0C00):
            self.char2id[chr(c)] = idx
            idx += 1
        # Digits (after normalizer converts subscripts)
        for d in "0123456789":
            self.char2id[d] = idx
            idx += 1
        # Common punctuation (important for prosody & pauses)
        for p in list(".,!?;:-'\""):
            self.char2id[p] = idx
            idx += 1
        self.vocab_size = idx

    def __len__(self):
        return len(self.ds)

    def text_to_ids(self, text):
        """Normalize with indic-nlp, then map each character to an ID."""
        text = normalize_tamil_text(text)
        ids = [self.char2id.get(ch, 0) for ch in text]
        ids = ids[:self.max_text_len]
        ids += [0] * (self.max_text_len - len(ids))
        return ids

    def __getitem__(self, idx):
        """
        Extract normalized text, raw audio, and compute mel-spectrogram.
        If a sample is corrupted, seamlessly skip to the next index.
        """
        total = len(self)
        for offset in range(total):
            actual_idx = (idx + offset) % total
            try:
                sample = self.ds[actual_idx]

                # --- 1. Extract Text (Supports Rasa, IndicVoices-R, Shrutilipi) ---
                text = (
                    sample.get("normalized")
                    or sample.get("text")
                    or sample.get("verbatim")
                    or ""
                )
                if not text.strip():
                    continue

                token_ids = torch.tensor(self.text_to_ids(text), dtype=torch.long)

                # --- 2. Extract Audio (Supports audio dict, AudioDecoder, audio_filepath) ---
                audio_data = sample.get("audio") or sample.get("audio_filepath")
                if audio_data is None:
                    continue

                if isinstance(audio_data, dict):
                    audio_array = np.array(audio_data["array"], dtype=np.float32)
                    sr = audio_data.get("sampling_rate", self.sr)
                elif hasattr(audio_data, "get_all_samples"):  # AudioDecoder object
                    audio_array = np.array(audio_data["array"], dtype=np.float32)
                    sr = getattr(audio_data, "sampling_rate", self.sr)
                elif isinstance(audio_data, str):  # file path string
                    audio_array, sr = librosa.load(audio_data, sr=self.sr)
                else:
                    audio_array = np.array(audio_data, dtype=np.float32)
                    sr = self.sr

                # Resample to 16kHz if needed
                if sr != self.sr:
                    audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=self.sr)

                # Skip extremely short audio (< 0.1s)
                if len(audio_array) < 1600:
                    continue

                # Pad or truncate raw waveform
                if len(audio_array) > self.max_audio_len:
                    audio_array = audio_array[:self.max_audio_len]
                else:
                    audio_array = np.pad(audio_array, (0, self.max_audio_len - len(audio_array)))

                audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

                # --- 3. Mel Spectrogram ---
                mel = librosa.feature.melspectrogram(
                    y=audio_array, sr=self.sr, n_fft=self.n_fft,
                    hop_length=self.hop_length, n_mels=self.mel_channels,
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                # Pad or truncate mel to fixed length (188 frames)
                if mel_db.shape[1] > self.max_mel_len:
                    mel_db = mel_db[:, :self.max_mel_len]
                else:
                    mel_db = np.pad(mel_db, ((0, 0), (0, self.max_mel_len - mel_db.shape[1])))

                mel_tensor = torch.tensor(mel_db, dtype=torch.float32)  # [80, max_mel_len]

                return token_ids, mel_tensor, audio_tensor

            except Exception:
                # Corrupted sample, skip to next index
                continue

        raise RuntimeError(f"Could not find any valid sample starting from index {idx}")


# Backward compatibility alias
ShrutilipiDataset = TamilTTSDataset


def load_dataset_splits(dataset_path, val_split=0.05):
    """
    Intelligently loads dataset splits from:
      1. Folder containing train.parquet + test.parquet (e.g. Rasa)
      2. Multi-file parquet directory (e.g. IndicVoices-R, Shrutilipi)
      3. Single parquet file
    """
    # Auto-resolve Kaggle input path variations
    search_paths = [
        dataset_path,
        dataset_path.replace("/datasets/ragunathravi/", "/"),
        f"/kaggle/input/{os.path.basename(dataset_path)}",
        f"/kaggle/input/datasets/ragunathravi/{os.path.basename(dataset_path)}"
    ]
    
    resolved_path = dataset_path
    for p in search_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    # Case 1: Folder containing dedicated train.parquet and test.parquet (Rasa)
    train_file = os.path.join(resolved_path, "train.parquet")
    test_file = os.path.join(resolved_path, "test.parquet")
    if os.path.isfile(train_file):
        print(f"Loading train split from: {train_file}")
        train_ds = load_dataset("parquet", data_files={"train": train_file})["train"]
        if os.path.isfile(test_file):
            print(f"Loading test split from: {test_file}")
            val_ds = load_dataset("parquet", data_files={"test": test_file})["test"]
        else:
            split = train_ds.train_test_split(test_size=val_split, seed=42)
            train_ds, val_ds = split["train"], split["test"]
        return train_ds, val_ds

    # Case 2: Direct file path
    if os.path.isfile(resolved_path):
        print(f"Loading single parquet file from: {resolved_path}")
        full_ds = load_dataset("parquet", data_files={"train": resolved_path})["train"]
        split = full_ds.train_test_split(test_size=val_split, seed=42)
        return split["train"], split["test"]

    # Case 3: Directory of parquet shards
    print(f"Loading parquet directory from: {resolved_path}")
    full_ds = load_dataset("parquet", data_dir=resolved_path, split="train")
    split = full_ds.train_test_split(test_size=val_split, seed=42)
    return split["train"], split["test"]


def build_dataloaders(cfg):
    """Build train and validation DataLoaders."""
    train_raw, val_raw = load_dataset_splits(cfg.dataset_dir, val_split=cfg.val_split)

    print(f"  Train samples: {len(train_raw)}")
    print(f"  Val samples  : {len(val_raw)}")

    train_ds = TamilTTSDataset(
        train_raw, max_audio_len=cfg.max_audio_len, max_text_len=cfg.max_text_len,
        mel_channels=cfg.mel_channels, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )
    val_ds = TamilTTSDataset(
        val_raw, max_audio_len=cfg.max_audio_len, max_text_len=cfg.max_text_len,
        mel_channels=cfg.mel_channels, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size if hasattr(cfg, 'batch_size') else cfg.per_gpu_batch,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size if hasattr(cfg, 'batch_size') else cfg.per_gpu_batch,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader
