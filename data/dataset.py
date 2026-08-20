import os
import glob
import re
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# Initialize IndicNLP Tamil Normalizer once at module level
_tamil_normalizer = IndicNormalizerFactory().get_normalizer("ta")

# Subscript digit mapping: ₀-₉ -> 0-9
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_tamil_text(text):
    """
    Complete Tamil text normalizer:
      1. IndicNLP Unicode normalizer for Tamil (canonical glyphs, vowel matras)
      2. Converts subscript digits (₀-₉) to standard ASCII digits (0-9)
      3. Cleans duplicate whitespace
    """
    if not isinstance(text, str):
        return ""
    text = _tamil_normalizer.normalize(text)
    text = text.translate(_SUBSCRIPT_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tamil_vocab(max_vocab=256):
    """
    Build character-level vocabulary for Tamil TTS.
    0: PAD, 1: SPACE, 2..: Tamil characters, digits, punctuation.
    """
    char2id = {" ": 1}
    idx = 2
    # Tamil Unicode block: 0x0B80 - 0x0BFF (128 code points)
    for c in range(0x0B80, 0x0C00):
        if idx < max_vocab:
            char2id[chr(c)] = idx
            idx += 1
    # Digits 0-9
    for d in "0123456789":
        if idx < max_vocab:
            char2id[d] = idx
            idx += 1
    # Common punctuation
    for p in list(".,!?;:-'\"()"):
        if idx < max_vocab:
            char2id[p] = idx
            idx += 1
    return char2id, max_vocab


class TamilTTSDataset(Dataset):
    """
    Universal dataset loader for Tamil TTS.
    Supports Rasa, IndicVoices-R, and Shrutilipi datasets seamlessly.
    Processes audio on-the-fly via multi-process DataLoaders.
    """
    def __init__(self, hf_dataset, cfg):
        self.ds = hf_dataset
        self.sr = cfg.sample_rate
        self.max_audio_len = cfg.max_audio_len
        self.max_text_len = cfg.max_text_len
        self.max_mel_len = cfg.max_mel_len
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.mel_channels = cfg.mel_channels

        self.char2id, self.vocab_size = build_tamil_vocab(max_vocab=getattr(cfg, "vocab_size", 256))

    def __len__(self):
        return len(self.ds)

    def text_to_ids(self, text):
        """Normalize Tamil text and convert to token IDs."""
        text = normalize_tamil_text(text)
        ids = [self.char2id.get(ch, 0) for ch in text]
        ids = [min(i, self.vocab_size - 1) for i in ids]
        ids = ids[:self.max_text_len]
        ids += [0] * (self.max_text_len - len(ids))
        return ids

    def __getitem__(self, idx):
        """
        Extracts normalized text, raw audio, and normalized mel spectrogram.
        Pads mel spectrogram with SILENCE (-1.0) rather than loud noise.
        """
        total = len(self)
        for offset in range(total):
            actual_idx = (idx + offset) % total
            try:
                sample = self.ds[actual_idx]

                # 1. Extract Text
                text = (
                    sample.get("normalized")
                    or sample.get("text")
                    or sample.get("verbatim")
                    or ""
                )
                if not text.strip():
                    continue

                token_ids = torch.tensor(self.text_to_ids(text), dtype=torch.long)

                # 2. Extract Audio
                audio_data = sample.get("audio") or sample.get("audio_filepath")
                if audio_data is None:
                    continue

                if isinstance(audio_data, dict):
                    audio_array = np.array(audio_data["array"], dtype=np.float32)
                    sr = audio_data.get("sampling_rate", self.sr)
                elif hasattr(audio_data, "get_all_samples"):
                    audio_array = np.array(audio_data["array"], dtype=np.float32)
                    sr = getattr(audio_data, "sampling_rate", self.sr)
                elif isinstance(audio_data, str):
                    audio_array, sr = librosa.load(audio_data, sr=self.sr)
                else:
                    audio_array = np.array(audio_data, dtype=np.float32)
                    sr = self.sr

                # Resample to 16kHz if needed
                if sr != self.sr:
                    audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=self.sr)

                # Skip extremely short clips (< 0.2s)
                if len(audio_array) < 3200:
                    continue

                # Compute Mel on clean unpadded audio first
                mel = librosa.feature.melspectrogram(
                    y=audio_array, sr=self.sr, n_fft=self.n_fft,
                    hop_length=self.hop_length, n_mels=self.mel_channels,
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)  # [-80.0, 0.0]

                # Normalize to [-1.0, 1.0] where -1.0 is silence and +1.0 is max loud
                mel_norm = np.clip((mel_db + 80.0) / 40.0 - 1.0, -1.0, 1.0)

                # Pad or truncate Mel Spectrogram with SILENCE (-1.0)
                if mel_norm.shape[1] > self.max_mel_len:
                    mel_norm = mel_norm[:, :self.max_mel_len]
                else:
                    mel_norm = np.pad(
                        mel_norm,
                        ((0, 0), (0, self.max_mel_len - mel_norm.shape[1])),
                        mode="constant",
                        constant_values=-1.0,  # CRITICAL: -1.0 is SILENCE
                    )

                # Pad or truncate raw audio with SILENCE (0.0)
                if len(audio_array) > self.max_audio_len:
                    audio_array = audio_array[:self.max_audio_len]
                else:
                    audio_array = np.pad(
                        audio_array,
                        (0, self.max_audio_len - len(audio_array)),
                        mode="constant",
                        constant_values=0.0,
                    )

                mel_tensor = torch.tensor(mel_norm, dtype=torch.float32)    # [80, max_mel_len]
                audio_tensor = torch.tensor(audio_array, dtype=torch.float32)  # [max_audio_len]

                return token_ids, mel_tensor, audio_tensor

            except Exception:
                continue

        raise RuntimeError(f"Could not find any valid sample starting from index {idx}")


# Backward compatibility alias
ShrutilipiDataset = TamilTTSDataset


def resolve_dataset_path(path):
    """Auto-resolve Kaggle dataset folder path variations."""
    if not isinstance(path, str):
        return path
    search_paths = [
        path,
        os.path.join(path, "data"),
        os.path.join("/kaggle/input", os.path.basename(path)),
        os.path.join("/kaggle/input/datasets", os.path.basename(path)),
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    return path


def load_single_dataset_splits(data_path, num_proc=None):
    """
    Loads train and validation/test splits from a dataset path using multi-processing.
    """
    if num_proc is None:
        num_proc = min(os.cpu_count() or 4, 8)

    resolved_path = resolve_dataset_path(data_path)
    print(f"  Loading dataset from: {resolved_path} (using {num_proc} CPU workers)")

    train_files = sorted(
        glob.glob(os.path.join(resolved_path, "*train*.parquet"))
        + glob.glob(os.path.join(resolved_path, "**", "*train*.parquet"), recursive=True)
    )
    test_files = sorted(
        glob.glob(os.path.join(resolved_path, "*test*.parquet"))
        + glob.glob(os.path.join(resolved_path, "**", "*test*.parquet"), recursive=True)
    )

    if train_files:
        print(f"    Found {len(train_files)} train parquet files")
        train_ds = load_dataset("parquet", data_files=train_files, split="train", num_proc=num_proc)
        if test_files:
            print(f"    Found {len(test_files)} test/val parquet files")
            val_ds = load_dataset("parquet", data_files=test_files, split="train", num_proc=num_proc)
        else:
            split_ds = train_ds.train_test_split(test_size=0.02, seed=42)
            train_ds, val_ds = split_ds["train"], split_ds["test"]
        return train_ds, val_ds

    all_parquets = sorted(
        glob.glob(os.path.join(resolved_path, "*.parquet"))
        + glob.glob(os.path.join(resolved_path, "**", "*.parquet"), recursive=True)
    )
    if all_parquets:
        print(f"    Found {len(all_parquets)} generic parquet files")
        full_ds = load_dataset("parquet", data_files=all_parquets, split="train", num_proc=num_proc)
        split_ds = full_ds.train_test_split(test_size=0.02, seed=42)
        return split_ds["train"], split_ds["test"]

    full_ds = load_dataset("parquet", data_dir=resolved_path, split="train", num_proc=num_proc)
    split_ds = full_ds.train_test_split(test_size=0.02, seed=42)
    return split_ds["train"], split_ds["test"]


def build_tamil_datasets(dataset_dirs, cfg, num_proc=None):
    """
    Builds combined Tamil TTS datasets with multi-core parallel parquet loading.
    """
    if num_proc is None:
        num_proc = min(os.cpu_count() or 4, 8)

    if isinstance(dataset_dirs, str):
        if "," in dataset_dirs:
            dirs = [d.strip() for d in dataset_dirs.split(",") if d.strip()]
        else:
            dirs = [dataset_dirs]
    elif isinstance(dataset_dirs, (list, tuple)):
        dirs = list(dataset_dirs)
    else:
        dirs = [str(dataset_dirs)]

    train_splits = []
    val_splits = []

    print("\n" + "=" * 60)
    print(f"  TamilTTS Fast Multi-Core Dataset Builder ({num_proc} CPU processes)")
    print("=" * 60)

    for d in dirs:
        try:
            train_hf, val_hf = load_single_dataset_splits(d, num_proc=num_proc)
            print(f"    ✓ Loaded: {len(train_hf):,} train, {len(val_hf):,} val samples from '{d}'")
            train_splits.append(TamilTTSDataset(train_hf, cfg))
            val_splits.append(TamilTTSDataset(val_hf, cfg))
        except Exception as e:
            print(f"    ⚠️ Warning: Could not load dataset from '{d}': {e}")

    if not train_splits:
        raise RuntimeError(f"No valid datasets could be loaded from: {dirs}")

    if len(train_splits) == 1:
        final_train = train_splits[0]
        final_val = val_splits[0]
    else:
        final_train = ConcatDataset(train_splits)
        final_val = ConcatDataset(val_splits)

    print(f"\n  🎉 Combined Dataset Ready:")
    print(f"  • Total Train Samples: {len(final_train):,}")
    print(f"  • Total Val Samples  : {len(final_val):,}")
    print("=" * 60 + "\n")

    return final_train, final_val


def build_dataloaders(cfg):
    """Builds PyTorch DataLoaders with persistent multi-processing workers."""
    train_ds, val_ds = build_tamil_datasets(cfg.dataset_dir, cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader
