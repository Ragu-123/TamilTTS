import os
import re
import glob
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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


def resolve_dataset_path(path):
    """Auto-resolve Kaggle input path variations."""
    if not isinstance(path, str):
        return path
    search_paths = [
        path,
        path.replace("/datasets/ragunathravi/", "/"),
        f"/kaggle/input/{os.path.basename(path)}",
        f"/kaggle/input/datasets/ragunathravi/{os.path.basename(path)}",
        f"/kaggle/input/{path}",
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    return path


def load_single_dataset_splits(dataset_path, val_split=0.05):
    """
    Loads train and test splits for a single dataset path.
    Supports:
      - Rasa (train.parquet / test.parquet)
      - IndicVoices-R (train-*.parquet / test-*.parquet)
      - Sharded parquet directory
      - Single parquet file
    """
    resolved_path = resolve_dataset_path(dataset_path)

    # 1. Check for explicit train-*.parquet and test-*.parquet files (IndicVoices-R & Rasa)
    if os.path.isdir(resolved_path):
        train_files = sorted(glob.glob(os.path.join(resolved_path, "*train*.parquet")))
        test_files  = sorted(glob.glob(os.path.join(resolved_path, "*test*.parquet")))

        if train_files:
            print(f"  ✓ Found {len(train_files)} train parquet file(s) in {resolved_path}")
            train_ds = load_dataset("parquet", data_files={"train": train_files})["train"]
            if test_files:
                print(f"  ✓ Found {len(test_files)} test parquet file(s) in {resolved_path}")
                val_ds = load_dataset("parquet", data_files={"test": test_files})["test"]
            else:
                split = train_ds.train_test_split(test_size=val_split, seed=42)
                train_ds, val_ds = split["train"], split["test"]
            return train_ds, val_ds

        # Shards without train/test in filename
        all_parquets = sorted(glob.glob(os.path.join(resolved_path, "*.parquet")))
        if all_parquets:
            print(f"  ✓ Found {len(all_parquets)} parquet file(s) in {resolved_path}")
            full_ds = load_dataset("parquet", data_files={"train": all_parquets})["train"]
            split = full_ds.train_test_split(test_size=val_split, seed=42)
            return split["train"], split["test"]

    # 2. Single Parquet file
    if os.path.isfile(resolved_path):
        print(f"  ✓ Loading single parquet file: {resolved_path}")
        full_ds = load_dataset("parquet", data_files={"train": resolved_path})["train"]
        split = full_ds.train_test_split(test_size=val_split, seed=42)
        return split["train"], split["test"]

    raise FileNotFoundError(f"Could not locate valid parquet files at: {dataset_path} (resolved: {resolved_path})")


def build_tamil_datasets(dataset_dirs, cfg):
    """
    Builds and combines datasets from one or multiple paths.
    Returns: (train_dataset, val_dataset) as PyTorch Datasets.
    """
    # Parse multiple comma-separated paths if provided
    if isinstance(dataset_dirs, str):
        paths = [p.strip() for p in dataset_dirs.split(",") if p.strip()]
    elif isinstance(dataset_dirs, (list, tuple)):
        paths = list(dataset_dirs)
    else:
        paths = [str(dataset_dirs)]

    train_datasets = []
    val_datasets = []

    print("=" * 60)
    print(f"  Loading & Combining {len(paths)} Tamil Dataset Source(s)...")
    print("=" * 60)

    for p in paths:
        try:
            train_raw, val_raw = load_single_dataset_splits(p, val_split=cfg.val_split)
            print(f"    --> Source: {os.path.basename(p)} | Train: {len(train_raw):,} | Val: {len(val_raw):,}")

            train_ds = TamilTTSDataset(
                train_raw, max_audio_len=cfg.max_audio_len, max_text_len=cfg.max_text_len,
                mel_channels=cfg.mel_channels, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            )
            val_ds = TamilTTSDataset(
                val_raw, max_audio_len=cfg.max_audio_len, max_text_len=cfg.max_text_len,
                mel_channels=cfg.mel_channels, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            )
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
        except Exception as e:
            print(f"    ⚠️ Warning: Could not load dataset at '{p}': {e}")

    if not train_datasets:
        raise RuntimeError("No valid Tamil datasets could be loaded! Please check dataset paths.")

    # Combine datasets
    if len(train_datasets) > 1:
        final_train_ds = ConcatDataset(train_datasets)
        final_val_ds   = ConcatDataset(val_datasets)
    else:
        final_train_ds = train_datasets[0]
        final_val_ds   = val_datasets[0]

    print(f"\n  🎉 Combined Dataset Total:")
    print(f"     • Total Train Samples: {len(final_train_ds):,}")
    print(f"     • Total Val Samples  : {len(final_val_ds):,}")
    print(f"     • Grand Total Audio  : {len(final_train_ds) + len(final_val_ds):,} studio samples (~75k+)")
    print("=" * 60)

    return final_train_ds, final_val_ds


def build_dataloaders(cfg):
    """Build train and validation DataLoaders (supports multi-dataset combination)."""
    train_ds, val_ds = build_tamil_datasets(cfg.dataset_dir, cfg)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size if hasattr(cfg, 'batch_size') else cfg.per_gpu_batch,
        shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size if hasattr(cfg, 'batch_size') else cfg.per_gpu_batch,
        shuffle=False, num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader
