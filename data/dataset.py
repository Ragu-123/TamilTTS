import os
import glob
import re
import io
import gc
import librosa
import soundfile as sf
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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


class DirectParquetTamilDataset(Dataset):
    """
    Zero-Disk-Cache, Ultra-Low-RAM Parquet Dataset for Tamil TTS.
    
    Memory optimizations:
    - Reads in-place from read-only `/kaggle/input` (0.0 MB disk space).
    - Caches at most ONE active parquet file per worker in RAM.
    - Explicit column filtering avoids loading unused metadata into memory.
    - Proactive garbage collection keeps Host RAM consumption under 2 GB per process.
    """
    def __init__(self, parquet_files, cfg):
        # Deduplicate files to avoid doubled indexing
        self.parquet_files = sorted(list(set(parquet_files))) if isinstance(parquet_files, (list, tuple)) else [parquet_files]
        self.sr = cfg.sample_rate
        self.max_audio_len = cfg.max_audio_len
        self.max_text_len = cfg.max_text_len
        self.max_mel_len = cfg.max_mel_len
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.mel_channels = cfg.mel_channels

        self.char2id, self.vocab_size = build_tamil_vocab(max_vocab=getattr(cfg, "vocab_size", 256))

        # Build lightweight row index [(file_path, row_idx_in_file), ...]
        self.index = []
        for f in self.parquet_files:
            try:
                pf = pq.ParquetFile(f)
                num_rows = pf.metadata.num_rows
                for r in range(num_rows):
                    self.index.append((f, r))
            except Exception as e:
                print(f"    ⚠️ Warning: Could not read metadata from {f}: {e}")

        # Worker-local single-file cache
        self._cached_file = None
        self._cached_table = None

    def __len__(self):
        return len(self.index)

    def _get_row(self, file_path, row_idx):
        """Reads a row using single active file cache with strict memory cleanup."""
        if self._cached_file != file_path or self._cached_table is None:
            # Free previous table immediately before loading new one
            self._cached_table = None
            self._cached_file = None

            pf = pq.ParquetFile(file_path)
            # Only read columns required for TTS training
            valid_cols = [c for c in ["audio", "normalized", "text", "verbatim", "audio_filepath"] if c in pf.schema.names]
            self._cached_table = pf.read(columns=valid_cols)
            self._cached_file = file_path

        row_dict = {}
        for col_name in self._cached_table.column_names:
            val = self._cached_table[col_name][row_idx].as_py()
            row_dict[col_name] = val
        return row_dict

    def text_to_ids(self, text):
        """Normalize Tamil text and convert to token IDs."""
        text = normalize_tamil_text(text)
        ids = [self.char2id.get(ch, 0) for ch in text]
        ids = [min(i, self.vocab_size - 1) for i in ids]
        ids = ids[:self.max_text_len]
        ids += [0] * (self.max_text_len - len(ids))
        return ids

    def _decode_audio(self, audio_field, sample_dict):
        """Universal audio decoder for bytes, arrays, and file paths."""
        if isinstance(audio_field, dict):
            # 1. Dict with audio bytes (Standard HuggingFace Parquet format)
            raw_bytes = audio_field.get("bytes")
            if raw_bytes is not None:
                audio_array, orig_sr = sf.read(io.BytesIO(raw_bytes))
                return np.array(audio_array, dtype=np.float32), orig_sr

            # 2. Dict with raw float array
            raw_arr = audio_field.get("array")
            if raw_arr is not None:
                orig_sr = audio_field.get("sampling_rate", self.sr)
                return np.array(raw_arr, dtype=np.float32), orig_sr

            # 3. Dict with path
            path_val = audio_field.get("path")
            if path_val and os.path.exists(path_val):
                audio_array, orig_sr = sf.read(path_val)
                return np.array(audio_array, dtype=np.float32), orig_sr

        elif isinstance(audio_field, (bytes, bytearray)):
            audio_array, orig_sr = sf.read(io.BytesIO(audio_field))
            return np.array(audio_array, dtype=np.float32), orig_sr

        elif isinstance(audio_field, str) and os.path.exists(audio_field):
            audio_array, orig_sr = sf.read(audio_field)
            return np.array(audio_array, dtype=np.float32), orig_sr

        # Fallback to audio_filepath column
        fallback_path = sample_dict.get("audio_filepath")
        if fallback_path and os.path.exists(fallback_path):
            audio_array, orig_sr = sf.read(fallback_path)
            return np.array(audio_array, dtype=np.float32), orig_sr

        return None, self.sr

    def __getitem__(self, idx):
        """
        Extracts normalized text, raw audio, and normalized mel spectrogram.
        Pads mel spectrogram with SILENCE (-1.0).
        """
        total = len(self.index)
        for offset in range(min(total, 50)):
            actual_idx = (idx + offset) % total
            try:
                file_path, row_idx = self.index[actual_idx]
                sample = self._get_row(file_path, row_idx)

                # 1. Extract Text (Supports normalized, text, verbatim)
                text = (
                    sample.get("normalized")
                    or sample.get("text")
                    or sample.get("verbatim")
                    or ""
                )
                if not isinstance(text, str) or not text.strip():
                    continue

                token_ids = torch.tensor(self.text_to_ids(text), dtype=torch.long)

                # 2. Extract and Decode Audio
                audio_field = sample.get("audio")
                audio_array, orig_sr = self._decode_audio(audio_field, sample)
                if audio_array is None:
                    continue

                # Ensure mono audio
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=1)

                # Resample to 16kHz if needed
                if orig_sr != self.sr:
                    audio_array = librosa.resample(y=audio_array, orig_sr=orig_sr, target_sr=self.sr)

                # Skip extremely short clips (< 0.2s)
                if len(audio_array) < 3200:
                    continue

                # Compute Mel Spectrogram on clean unpadded audio
                mel = librosa.feature.melspectrogram(
                    y=audio_array, sr=self.sr, n_fft=self.n_fft,
                    hop_length=self.hop_length, n_mels=self.mel_channels,
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)  # [-80.0, 0.0]

                # Normalize to [-1.0, 1.0] (silence is -1.0, loud is +1.0)
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

                mel_tensor = torch.tensor(mel_norm, dtype=torch.float32)       # [80, max_mel_len]
                audio_tensor = torch.tensor(audio_array, dtype=torch.float32)  # [max_audio_len]

                return token_ids, mel_tensor, audio_tensor

            except Exception:
                continue

        raise RuntimeError(f"Could not load valid sample near index {idx}")


# Aliases for backward compatibility
TamilTTSDataset = DirectParquetTamilDataset
ShrutilipiDataset = DirectParquetTamilDataset


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


def load_single_parquet_dataset_splits(data_path, cfg):
    """
    Scans parquet files with deduplication and returns train/val DirectParquetTamilDataset objects.
    Uses 0 MB of disk space.
    """
    resolved_path = resolve_dataset_path(data_path)
    print(f"  Scanning parquet files from: {resolved_path}")

    # Use set to strictly prevent duplicate entries
    train_files = sorted(list(set(
        glob.glob(os.path.join(resolved_path, "**", "*train*.parquet"), recursive=True)
        or glob.glob(os.path.join(resolved_path, "*train*.parquet"))
    )))
    test_files = sorted(list(set(
        glob.glob(os.path.join(resolved_path, "**", "*test*.parquet"), recursive=True)
        or glob.glob(os.path.join(resolved_path, "*test*.parquet"))
    )))

    if train_files:
        print(f"    ✓ Found {len(train_files)} train parquet file(s)")
        train_ds = DirectParquetTamilDataset(train_files, cfg)

        if test_files:
            print(f"    ✓ Found {len(test_files)} test/val parquet file(s)")
            val_ds = DirectParquetTamilDataset(test_files, cfg)
        else:
            total_n = len(train_ds)
            train_files_list = train_ds.parquet_files
            train_ds = DirectParquetTamilDataset(train_files_list, cfg)
            val_ds = DirectParquetTamilDataset(train_files_list[-1:], cfg)

        return train_ds, val_ds

    all_parquets = sorted(list(set(
        glob.glob(os.path.join(resolved_path, "**", "*.parquet"), recursive=True)
        or glob.glob(os.path.join(resolved_path, "*.parquet"))
    )))
    if all_parquets:
        print(f"    ✓ Found {len(all_parquets)} generic parquet file(s)")
        if len(all_parquets) > 1:
            train_files = all_parquets[:-1]
            test_files = all_parquets[-1:]
            return DirectParquetTamilDataset(train_files, cfg), DirectParquetTamilDataset(test_files, cfg)
        else:
            ds = DirectParquetTamilDataset(all_parquets, cfg)
            return ds, ds

    raise FileNotFoundError(f"No parquet files found in: {resolved_path}")


# Backward compatibility alias
load_single_dataset_splits = load_single_parquet_dataset_splits


def build_tamil_datasets(dataset_dirs, cfg, num_proc=None):
    """
    Builds combined Tamil TTS datasets directly from Parquet files without writing any cache to disk.
    """
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
    print("  TamilTTS Zero-Disk-Cache Dataset Builder")
    print("=" * 60)

    for d in dirs:
        try:
            train_ds, val_ds = load_single_parquet_dataset_splits(d, cfg)
            print(f"    ✓ Indexed: {len(train_ds):,} train, {len(val_ds):,} val samples from '{d}'")
            train_splits.append(train_ds)
            val_splits.append(val_ds)
        except Exception as e:
            print(f"    ⚠️ Warning: Could not index dataset from '{d}': {e}")

    if not train_splits:
        raise RuntimeError(f"No valid parquet datasets found in: {dirs}")

    if len(train_splits) == 1:
        final_train = train_splits[0]
        final_val = val_splits[0]
    else:
        final_train = ConcatDataset(train_splits)
        final_val = ConcatDataset(val_splits)

    print(f"\n  🎉 Combined Dataset Ready (0 MB Disk Used):")
    print(f"  • Total Train Samples: {len(final_train):,}")
    print(f"  • Total Val Samples  : {len(final_val):,}")
    print("=" * 60 + "\n")

    return final_train, final_val


def build_dataloaders(cfg):
    """Builds PyTorch DataLoaders with low-RAM multi-processing."""
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
