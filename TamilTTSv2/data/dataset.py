"""
High-Throughput Streaming Parquet Dataset for TamilTTSv2 (FastPitch / RAD-TTS Standard)
========================================================================================
- 100% MFA ground-truth durations: samples without a durations_dict entry are skipped.
- Exact G2G Akshara Tokenization from duration entries (no text re-segmentation).
- Per-sample prosody features: utterance-normalized log-F0, voicing mask, log-energy.
- Dynamic Batching & Collation: sequences padded only to the max length in the batch.
- Sample Rate: 22,050 Hz (exact match for pre-trained HiFi-GAN V1).
"""
import glob
import io
import os
import random

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torchaudio.functional as AF
from torch.utils.data import Dataset

from data.audio_features import MelExtractor, extract_energy, extract_f0
from preprocess.g2g import TAMIL_G2G_TOKENS, VOCAB_SIZE


def build_tamil_vocab():
    """Build the G2G token-to-id mapping (vocab_size=384 compatible)."""
    char2id = {tok: idx for idx, tok in enumerate(TAMIL_G2G_TOKENS)}
    return char2id, VOCAB_SIZE


class DirectParquetTamilDataset(Dataset):
    """
    Row-Group Level Streaming Parquet Dataset for Tamil TTS with 100% MFA Ground Truth.

    __getitem__ returns:
        (token_ids[Tt] long, text_len int, mel_log[80, Tm] float,
         mel_len int, audio[Ta] float, audio_len int, gt_dur[Tt] float32,
         log_f0[Tm], voiced_mask[Tm], energy[Tm])
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

        self.char2id, self.vocab_size = build_tamil_vocab()
        self.g2g_set = set(TAMIL_G2G_TOKENS)

        durations_file = getattr(cfg, "durations_file", None)
        if not durations_file:
            raise ValueError(
                "DirectParquetTamilDataset requires cfg.durations_file: "
                "fabricated uniform durations are not supported in v2."
            )
        if not os.path.exists(durations_file):
            raise FileNotFoundError(f"FATAL ERROR: Configured durations_file '{durations_file}' not found on disk!")
        try:
            self.durations_dict = torch.load(durations_file, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"FATAL ERROR: Failed to load durations file '{durations_file}': {e}")
        valid_keys = set(self.durations_dict.keys())
        print(f"  ✓ Loaded {len(self.durations_dict):,} ground-truth MFA durations from '{durations_file}'")

        self.index = []
        for f in self.parquet_files:
            try:
                pf = pq.ParquetFile(f, memory_map=True)
                base_f = os.path.basename(f)
                for rg_idx in range(pf.num_row_groups):
                    num_rows = pf.metadata.row_group(rg_idx).num_rows
                    for r in range(num_rows):
                        key = f"{base_f}_rg{rg_idx}_r{r}"
                        if key in valid_keys:
                            self.index.append((f, rg_idx, r, key))
            except Exception as e:
                print(f"    ⚠️ Warning: Could not read metadata from {f}: {e}")

        if len(self.index) == 0:
            raise RuntimeError(
                f"FATAL ERROR: Zero dataset samples matched the keys in '{durations_file}'!"
            )
        print(f"  ✓ Strict MFA Filtering: Training on {len(self.index):,} verified aligned samples (100% G2G ground-truth).")

        self._cached_key = None
        self._cached_table = None
        self._mel_processor = None
        self._fail_count = 0

    def __len__(self):
        return len(self.index)

    def _get_mel_processor(self):
        if self._mel_processor is None:
            self._mel_processor = MelExtractor(
                sample_rate=self.sr, n_fft=self.n_fft, hop_length=self.hop_length,
                n_mels=self.mel_channels, fmin=self.f_min, fmax=self.f_max
            )
        return self._mel_processor

    def _get_row(self, file_path, rg_idx, row_in_rg):
        cache_key = (file_path, rg_idx)
        if self._cached_key != cache_key or self._cached_table is None:
            self._cached_table = None
            pf = pq.ParquetFile(file_path, memory_map=True)
            self._cached_table = pf.read_row_group(rg_idx)
            self._cached_key = cache_key

        row_dict = {}
        for col_name in self._cached_table.column_names:
            row_dict[col_name] = self._cached_table[col_name][row_in_rg].as_py()
        return row_dict

    @staticmethod
    def _fit_to_len(t, n):
        if t.numel() == n:
            return t
        if t.numel() > n:
            return t[:n]
        return torch.cat([t, torch.zeros(n - t.numel(), dtype=t.dtype)])

    def _register_failure(self):
        self._fail_count += 1
        if self._fail_count % 200 == 0:
            print(f"    ⚠️ Warning: dataset has skipped {self._fail_count} failed samples so far.")

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
        total = len(self.index)
        max_attempts = total * 2
        consecutive_failures = 0

        for offset in range(max_attempts):
            actual_idx = (idx + offset) % total
            try:
                file_path, rg_idx, row_in_rg, sample_key = self.index[actual_idx]
                sample = self._get_row(file_path, rg_idx, row_in_rg)

                entry = self.durations_dict.get(sample_key)
                if entry is None:
                    self._register_failure()
                    consecutive_failures += 1
                    continue

                toks = entry.get("tokens", [])
                durs = entry.get("durations", [])
                if len(toks) < 2 or len(toks) > self.max_text_len or len(toks) != len(durs):
                    self._register_failure()
                    consecutive_failures += 1
                    continue

                token_ids = torch.tensor([self.char2id.get(t, 2) for t in toks], dtype=torch.long)
                text_len = len(token_ids)
                gt_dur = torch.tensor([float(d) for d in durs], dtype=torch.float32)

                audio_array, orig_sr = self._decode_audio(sample)
                if audio_array is None or len(audio_array) == 0:
                    self._register_failure()
                    consecutive_failures += 1
                    continue

                audio_array = np.asarray(audio_array, dtype=np.float32)
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=1)

                if orig_sr != self.sr:
                    resampled = AF.resample(torch.from_numpy(audio_array).unsqueeze(0), orig_sr, self.sr)
                    audio_array = resampled.squeeze(0).numpy()

                audio_len = len(audio_array)
                if audio_len < self.min_audio_len or audio_len > self.max_audio_len:
                    self._register_failure()
                    consecutive_failures += 1
                    continue

                audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
                mel_log = self._get_mel_processor()(audio_tensor).squeeze(0)
                mel_len = mel_log.shape[1]

                if mel_len < 4 or mel_len < text_len:
                    self._register_failure()
                    consecutive_failures += 1
                    continue

                dur_sum = int(round(gt_dur.sum().item()))
                diff = mel_len - dur_sum
                if abs(diff) <= max(10, int(0.05 * mel_len)):
                    target_idx = -1 if gt_dur[-1] >= gt_dur[0] else 0
                    gt_dur[target_idx] = max(1.0, gt_dur[target_idx].item() + diff)
                else:
                    self._register_failure()
                    consecutive_failures += 1
                    continue

                if len(gt_dur) != text_len or int(round(gt_dur.sum().item())) != mel_len:
                    self._register_failure()
                    consecutive_failures += 1
                    continue

                log_f0, voiced_mask = extract_f0(
                    audio_tensor, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft
                )
                energy = extract_energy(audio_tensor, hop_length=self.hop_length, n_fft=self.n_fft)
                log_f0 = self._fit_to_len(log_f0, mel_len)
                voiced_mask = self._fit_to_len(voiced_mask, mel_len)
                energy = self._fit_to_len(energy, mel_len)

                consecutive_failures = 0
                return (token_ids, text_len, mel_log, mel_len, audio_tensor, audio_len,
                        gt_dur, log_f0, voiced_mask, energy)

            except Exception as e:
                self._register_failure()
                consecutive_failures += 1
                if consecutive_failures >= max_attempts:
                    raise RuntimeError("Too many consecutive dataset failures") from e
                continue

        raise RuntimeError("Too many consecutive dataset failures")


def tamil_tts_collate_fn(batch):
    """
    Dynamic Batch Collation: pads all sequences to the batch maximum and returns a dict.

    Returns:
        Dict[str, Tensor]:
            tokens     LongTensor   [B, Tt]
            token_lens LongTensor   [B]
            mel        FloatTensor  [B, 80, Tm]   (pad fill -4.0)
            mel_lens   LongTensor   [B]
            audio      FloatTensor  [B, Ta]
            audio_lens LongTensor   [B]
            gt_dur     FloatTensor  [B, Tt]       (0 on pad positions)
            log_f0     FloatTensor  [B, Tm]       (0 on unvoiced/pad)
            voiced     FloatTensor  [B, Tm]
            energy     FloatTensor  [B, Tm]       (0 on pad)
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    B = len(batch)
    token_lens = torch.tensor([b[1] for b in batch], dtype=torch.long)
    mel_lens = torch.tensor([b[3] for b in batch], dtype=torch.long)
    audio_lens = torch.tensor([b[5] for b in batch], dtype=torch.long)

    max_text_len = int(token_lens.max().item())
    max_mel_len = int(mel_lens.max().item())
    max_audio_len = int(audio_lens.max().item())

    out = {
        "tokens": torch.zeros(B, max_text_len, dtype=torch.long),
        "token_lens": token_lens,
        "mel": torch.full((B, batch[0][2].shape[0], max_mel_len), -4.0, dtype=torch.float32),
        "mel_lens": mel_lens,
        "audio": torch.zeros(B, max_audio_len, dtype=torch.float32),
        "audio_lens": audio_lens,
        "gt_dur": torch.zeros(B, max_text_len, dtype=torch.float32),
        "log_f0": torch.zeros(B, max_mel_len, dtype=torch.float32),
        "voiced": torch.zeros(B, max_mel_len, dtype=torch.float32),
        "energy": torch.zeros(B, max_mel_len, dtype=torch.float32),
    }

    for i, item in enumerate(batch):
        toks, t_len, mel, m_len, aud, a_len, gt_dur, log_f0, voiced, energy = item
        t_len = min(t_len, toks.shape[0])
        m_len = min(m_len, mel.shape[1], max_mel_len)
        a_len = min(a_len, aud.shape[0])
        out["tokens"][i, :t_len] = toks[:t_len]
        out["mel"][i, :, :m_len] = mel[:, :m_len]
        out["audio"][i, :a_len] = aud[:a_len]
        out["gt_dur"][i, :t_len] = gt_dur[:t_len]
        out["log_f0"][i, :m_len] = log_f0[:m_len]
        out["voiced"][i, :m_len] = voiced[:m_len]
        out["energy"][i, :m_len] = energy[:m_len]

    return out


TamilTTSDataset = DirectParquetTamilDataset


def build_tamil_datasets(dataset_dirs, cfg, val_split=0.02):
    """
    Build deterministic train and validation datasets from parquet directories.
    Globbed recursively with a fixed seed-42 random split into Subsets.

    Args:
        dataset_dirs (str | List[str]): Directories containing *.parquet files.
        cfg: Config object consumed by DirectParquetTamilDataset.
        val_split (float): Fraction of samples held out for validation.
    Returns:
        Tuple[Subset, Subset]: (train_ds, val_ds).
    """
    if isinstance(dataset_dirs, str):
        dataset_dirs = [dataset_dirs]

    parquet_files = []
    for d in dataset_dirs:
        found = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
        parquet_files.extend(found)

    parquet_files = sorted(list(set(parquet_files)))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dirs}")

    full_ds = DirectParquetTamilDataset(parquet_files, cfg)

    total_samples = len(full_ds)
    val_size = max(int(total_samples * val_split), 20)
    train_size = total_samples - val_size

    indices = list(range(total_samples))
    rng = random.Random(42)
    rng.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_ds = torch.utils.data.Subset(full_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_ds, val_indices)

    return train_ds, val_ds
