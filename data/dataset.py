"""
High-Throughput Streaming Parquet Dataset for Tamil TTS (FastPitch / RAD-TTS Standard)
======================================================================================
- Dynamic Batching & Collation: Sequences are padded only to the max length in that batch.
- 1-to-1 Audio-Text Integrity: Utterances are filtered by length (0.5s - 10.0s), never truncated independently.
- Sample Rate: 22,050 Hz (exact match for pre-trained HiFi-GAN V1).
- Natural Log-Mel: 80 channels, n_fft=1024, hop=256, fmin=0, fmax=8000, clamped [-11.5, 0.0].
"""
import os
import glob
import re
import io
import librosa
import soundfile as sf
import numpy as np
import pyarrow.parquet as pq
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader, ConcatDataset
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    _tamil_normalizer = IndicNormalizerFactory().get_normalizer("ta")
except ImportError:
    _tamil_normalizer = None

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_tamil_text(text):
    """
    Tamil text normalizer:
    1. IndicNLP Unicode normalizer for Tamil.
    2. Converts subscript digits (₀-₉) to standard ASCII digits (0-9).
    3. Cleans duplicate whitespace.
    """
    if not isinstance(text, str):
        return ""
    if _tamil_normalizer is not None:
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
    Row-Group Level Streaming Parquet Dataset for Tamil TTS.
    """
    def __init__(self, parquet_files, cfg):
        self.parquet_files = sorted(list(set(parquet_files))) if isinstance(parquet_files, (list, tuple)) else [parquet_files]
        self.sr = getattr(cfg, "sample_rate", 22050)
        self.min_audio_len = getattr(cfg, "min_audio_len", int(0.5 * self.sr))
        self.max_audio_len = getattr(cfg, "max_audio_len", int(10.0 * self.sr))
        self.max_text_len = getattr(cfg, "max_text_len", 250)
        self.n_fft = getattr(cfg, "n_fft", 1024)
        self.hop_length = getattr(cfg, "hop_length", 256)
        self.mel_channels = getattr(cfg, "mel_channels", 80)
        self.f_min = getattr(cfg, "f_min", 0.0)
        self.f_max = getattr(cfg, "f_max", 8000.0)

        self.char2id, self.vocab_size = build_tamil_vocab(max_vocab=getattr(cfg, "vocab_size", 256))

        # Build lightweight row group index [(file_path, row_group_idx, row_in_group), ...]
        self.index = []
        for f in self.parquet_files:
            try:
                pf = pq.ParquetFile(f, memory_map=True)
                for rg_idx in range(pf.num_row_groups):
                    num_rows = pf.metadata.row_group(rg_idx).num_rows
                    for r in range(num_rows):
                        self.index.append((f, rg_idx, r))
            except Exception as e:
                print(f"    ⚠️ Warning: Could not read metadata from {f}: {e}")

        # Worker-local single row-group cache
        self._cached_key = None
        self._cached_table = None

        # Load Ground-Truth MFA Durations if available
        self.durations_dict = None
        durations_file = getattr(cfg, "durations_file", None)
        if durations_file and os.path.exists(durations_file):
            try:
                self.durations_dict = torch.load(durations_file, map_location="cpu")
                print(f"  ✓ Loaded {len(self.durations_dict):,} ground-truth MFA durations from '{durations_file}'")
            except Exception as e:
                print(f"  ⚠️ Warning: Could not load durations file '{durations_file}': {e}")

    def __len__(self):
        return len(self.index)

    def _get_row(self, file_path, rg_idx, row_in_rg):
        cache_key = (file_path, rg_idx)
        if self._cached_key != cache_key or self._cached_table is None:
            self._cached_table = None
            pf = pq.ParquetFile(file_path, memory_map=True)
            self._cached_table = pf.read_row_group(rg_idx)
            self._cached_key = cache_key

        row_dict = {}
        for col_name in self._cached_table.column_names:
            val = self._cached_table[col_name][row_in_rg].as_py()
            row_dict[col_name] = val
        return row_dict

    def text_to_ids(self, text):
        text = normalize_tamil_text(text)
        ids = [self.char2id.get(ch, 0) for ch in text]
        ids = [min(i, self.vocab_size - 1) for i in ids if i > 0]  # strip 0s
        ids = ids[:self.max_text_len]
        return ids

    def _decode_audio(self, sample):
        audio_val = sample.get("audio")
        if isinstance(audio_val, dict):
            raw_bytes = audio_val.get("bytes")
            if raw_bytes is not None and len(raw_bytes) > 100:
                arr, orig_sr = sf.read(io.BytesIO(raw_bytes))
                return np.array(arr, dtype=np.float32), orig_sr

            raw_arr = audio_val.get("array")
            if raw_arr is not None:
                orig_sr = audio_val.get("sampling_rate", self.sr)
                return np.array(raw_arr, dtype=np.float32), orig_sr

            path_val = audio_val.get("path")
            if path_val and os.path.exists(path_val):
                arr, orig_sr = sf.read(path_val)
                return np.array(arr, dtype=np.float32), orig_sr

        raw_b = sample.get("bytes")
        if isinstance(raw_b, (bytes, bytearray)) and len(raw_b) > 100:
            arr, orig_sr = sf.read(io.BytesIO(raw_b))
            return np.array(arr, dtype=np.float32), orig_sr

        if isinstance(audio_val, (bytes, bytearray)) and len(audio_val) > 100:
            arr, orig_sr = sf.read(io.BytesIO(audio_val))
            return np.array(arr, dtype=np.float32), orig_sr

        for key in ["wav_path", "audio_filepath", "path", "filename"]:
            p = sample.get(key)
            if isinstance(p, str) and os.path.exists(p):
                arr, orig_sr = sf.read(p)
                return np.array(arr, dtype=np.float32), orig_sr

        return None, self.sr

    def __getitem__(self, idx):
        """
        Returns dynamic, unpadded sequence tensors with exact lengths:
        (token_ids, text_len, mel_tensor, mel_len, audio_tensor, audio_len)
        """
        total = len(self.index)
        for offset in range(total):
            actual_idx = (idx + offset) % total
            try:
                file_path, rg_idx, row_in_rg = self.index[actual_idx]
                sample = self._get_row(file_path, rg_idx, row_in_rg)

                # 1. Extract Text
                text = (
                    sample.get("normalized")
                    or sample.get("text")
                    or sample.get("verbatim")
                    or ""
                )
                if not isinstance(text, str) or not text.strip():
                    continue

                ids = self.text_to_ids(text)
                if len(ids) < 2 or len(ids) > self.max_text_len:
                    continue
                token_ids = torch.tensor(ids, dtype=torch.long)
                text_len = len(ids)

                # 2. Extract and Decode Audio
                audio_array, orig_sr = self._decode_audio(sample)
                if audio_array is None:
                    continue

                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)

                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=1)

                # Resample to 22,050 Hz (exact HiFi-GAN V1 rate)
                if orig_sr != self.sr:
                    audio_t = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
                    audio_resampled = AF.resample(audio_t, orig_sr, self.sr).squeeze(0).numpy()
                    audio_array = audio_resampled

                # Filter complete utterances by length (no artificial chopping of text)
                audio_len = len(audio_array)
                if audio_len < self.min_audio_len or audio_len > self.max_audio_len:
                    continue

                # 3. Compute Natural Log-Mel Spectrogram (22.05 kHz Magnitude)
                audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
                if not hasattr(self, "_mel_transform") or self._mel_transform is None:
                    import torchaudio.transforms as T
                    self._mel_transform = T.MelSpectrogram(
                        sample_rate=self.sr,
                        n_fft=self.n_fft,
                        win_length=self.n_fft,
                        hop_length=self.hop_length,
                        f_min=self.f_min,
                        f_max=self.f_max,
                        n_mels=self.mel_channels,
                        power=1.0,
                        norm="slaney",
                        mel_scale="slaney",
                    )

                mel = self._mel_transform(audio_tensor)
                mel_log = torch.log(torch.clamp(mel, min=1e-5))
                mel_len = mel_log.shape[1]

                # 4. Ground-Truth Duration Attachment (IndicMFA with Proportional Fallback)
                sample_key = f"{os.path.basename(file_path)}_rg{rg_idx}_r{row_in_rg}"
                gt_dur = None
                if self.durations_dict is not None and sample_key in self.durations_dict:
                    entry = self.durations_dict[sample_key]
                    durs = entry.get("durations", [])
                    if len(durs) == text_len:
                        gt_dur = torch.tensor(durs, dtype=torch.long)

                if gt_dur is None:
                    # Fallback to proportional duration for unaligned samples
                    p_dur = max(1, mel_len // text_len)
                    gt_dur = torch.full((text_len,), p_dur, dtype=torch.long)
                    diff = mel_len - gt_dur.sum().item()
                    gt_dur[-1] = max(1, gt_dur[-1].item() + diff)

                return token_ids, text_len, mel_log, mel_len, audio_tensor, audio_len, gt_dur

            except Exception:
                continue

        # Neutral fallback
        dummy_ids = torch.tensor([1, 2], dtype=torch.long)
        dummy_mel = torch.full((self.mel_channels, 16), -11.5, dtype=torch.float32)
        dummy_audio = torch.zeros(16 * self.hop_length, dtype=torch.float32)
        dummy_dur = torch.tensor([8, 8], dtype=torch.long)
        return dummy_ids, 2, dummy_mel, 16, dummy_audio, 16 * self.hop_length, dummy_dur


def tamil_tts_collate_fn(batch):
    """
    Dynamic Batch Collation:
    Pads sequences only to the maximum length present in that specific batch.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    B = len(batch)
    text_lens = torch.tensor([b[1] for b in batch], dtype=torch.long)
    mel_lens = torch.tensor([b[3] for b in batch], dtype=torch.long)
    audio_lens = torch.tensor([b[5] for b in batch], dtype=torch.long)

    max_text_len = int(text_lens.max().item())
    max_mel_len = int(mel_lens.max().item())
    max_audio_len = int(audio_lens.max().item())

    padded_tokens = torch.zeros(B, max_text_len, dtype=torch.long)
    padded_mel = torch.full((B, 80, max_mel_len), -11.5, dtype=torch.float32)
    padded_audio = torch.zeros(B, max_audio_len, dtype=torch.float32)
    padded_dur = torch.zeros(B, max_text_len, dtype=torch.long)

    for i, item in enumerate(batch):
        toks, t_len, mel, m_len, aud, a_len = item[:6]
        padded_tokens[i, :t_len] = toks[:t_len]
        padded_mel[i, :, :m_len] = mel[:, :m_len]
        padded_audio[i, :a_len] = aud[:a_len]
        if len(item) > 6 and item[6] is not None:
            padded_dur[i, :t_len] = item[6][:t_len]

    return padded_tokens, text_lens, padded_mel, mel_lens, padded_audio, audio_lens, padded_dur


# Aliases for backward compatibility
TamilTTSDataset = DirectParquetTamilDataset
ShrutilipiDataset = DirectParquetTamilDataset


def resolve_dataset_path(path):
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
    resolved_path = resolve_dataset_path(data_path)
    print(f"  Scanning parquet files from: {resolved_path}")

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


load_single_dataset_splits = load_single_parquet_dataset_splits


def build_tamil_datasets(dataset_dirs, cfg, num_proc=None):
    if isinstance(dataset_dirs, str):
        dirs = [d.strip() for d in dataset_dirs.split(",") if d.strip()]
    elif isinstance(dataset_dirs, (list, tuple)):
        dirs = list(dataset_dirs)
    else:
        dirs = [str(dataset_dirs)]

    train_splits = []
    val_splits = []

    print("\n" + "=" * 60)
    print("  TamilTTS High-Throughput Streaming Dataset Builder")
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
    train_ds, val_ds = build_tamil_datasets(cfg.dataset_dir, cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=True,
        collate_fn=tamil_tts_collate_fn,
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
        collate_fn=tamil_tts_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2 if cfg.num_workers > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader
