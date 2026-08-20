"""
TamilTTS Inference Script
=========================
Generates Tamil speech audio from text using a trained TamilTTS checkpoint.

Usage (Kaggle):
    %cd /kaggle/working/TamilTTS
    !python inference.py --text "வணக்கம், நான் தமிழில் பேசுகிறேன்." --checkpoint /path/to/best.pt

Usage (local):
    python inference.py --text "வணக்கம்" --checkpoint ./checkpoints/best.pt --output output.wav
"""
import argparse
import os
import re
import torch
import numpy as np
import soundfile as sf
from config import Config
from models import TamilTTS
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# ---- Text Processing (same as training) ----
_tamil_normalizer = IndicNormalizerFactory().get_normalizer("ta")
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_tamil_text(text):
    if not isinstance(text, str):
        return ""
    text = _tamil_normalizer.normalize(text)
    text = text.translate(_SUBSCRIPT_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_char2id(max_vocab_size=128):
    """
    Build character vocabulary with strict index bounding to prevent CUDA out-of-bounds.
    """
    char2id = {" ": 1}  # 0=PAD, 1=SPACE
    idx = 2
    # Tamil Unicode block (0x0B80 - 0x0BFF)
    for c in range(0x0B80, 0x0C00):
        if idx < max_vocab_size:
            char2id[chr(c)] = idx
            idx += 1
    # Digits
    for d in "0123456789":
        if idx < max_vocab_size:
            char2id[d] = idx
            idx += 1
    # Punctuation
    for p in list(".,!?;:-'\""):
        if idx < max_vocab_size:
            char2id[p] = idx
            idx += 1
    return char2id


def text_to_ids(text, char2id, max_text_len=200, max_vocab_size=128):
    """Convert Tamil text to token IDs safely bounded by vocab_size."""
    text = normalize_tamil_text(text)
    ids = []
    for ch in text:
        token_id = char2id.get(ch, 0)
        # Safety clamp to prevent CUDA gather kernel asserts
        if token_id >= max_vocab_size:
            token_id = 0
        ids.append(token_id)

    ids = ids[:max_text_len]
    ids += [0] * (max_text_len - len(ids))
    return ids


@torch.no_grad()
def synthesize(model, text, char2id, device, vocab_size=128, ref_mel=None, max_text_len=200):
    """
    Generate speech audio from Tamil text.
    """
    # 1. Tokenize text with safe vocab clamping
    token_ids = text_to_ids(text, char2id, max_text_len, max_vocab_size=vocab_size)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)  # [1, T_text]

    # 2. Reference mel for style extraction
    if ref_mel is None:
        ref_mel = torch.zeros(1, 80, 50, device=device)  # [1, 80, 50]
    elif ref_mel.dim() == 2:
        ref_mel = ref_mel.unsqueeze(0)
    ref_mel = ref_mel.to(device)

    # 3. Model forward pass
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()

    audio, mel_pred, dur_pred = eval_model(tokens, ref_mel, target_mel_len=None)

    # 4. Convert to numpy
    audio_np = audio.squeeze(0).cpu().numpy()  # [T_audio]
    mel_np = mel_pred.squeeze(0).cpu().numpy()  # [T_mel, 80]

    # 5. Trim trailing silence based on predicted durations
    dur_np = dur_pred.squeeze(0).cpu().numpy()
    non_pad = (np.array(token_ids) != 0)
    total_dur_frames = int(np.sum(np.round(np.clip(dur_np[non_pad], 0, None))))
    if total_dur_frames > 0:
        hop_length = 256
        trim_samples = min(total_dur_frames * hop_length, len(audio_np))
        if trim_samples > 1600:
            audio_np = audio_np[:trim_samples]

    # Normalize volume to avoid clipping
    max_val = np.abs(audio_np).max()
    if max_val > 0.01:
        audio_np = audio_np / max_val * 0.95

    return audio_np, mel_np


def main():
    parser = argparse.ArgumentParser(description="TamilTTS Inference")
    parser.add_argument("--text", type=str, required=True, help="Tamil text to synthesize")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (best.pt)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV file path")
    parser.add_argument("--ref_audio", type=str, default=None, help="Optional: reference audio WAV for voice cloning")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    print("=" * 60)
    print("  TamilTTS Inference Pipeline")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Text       : {args.text}")

    # 1. Inspect checkpoint to auto-match vocab_size
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    emb_weight = ckpt["model_state_dict"].get("text_encoder.embedding.weight")
    if emb_weight is not None:
        ckpt_vocab_size = emb_weight.shape[0]
        cfg.vocab_size = ckpt_vocab_size
        print(f"  Vocab Size : {ckpt_vocab_size} (Auto-matched from checkpoint)")
    else:
        ckpt_vocab_size = cfg.vocab_size

    # 2. Instantiate and load model
    model = TamilTTS(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded from: Step {ckpt.get('step', 0)}, Loss {ckpt.get('loss', 0.0):.4f}")

    # 3. Build vocabulary bounded by checkpoint vocab size
    char2id = build_char2id(max_vocab_size=ckpt_vocab_size)

    # 4. Optional voice cloning reference audio
    ref_mel = None
    if args.ref_audio and os.path.exists(args.ref_audio):
        import librosa
        audio_ref, sr = librosa.load(args.ref_audio, sr=cfg.sample_rate)
        mel = librosa.feature.melspectrogram(
            y=audio_ref, sr=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, n_mels=cfg.mel_channels,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        ref_mel = torch.tensor(mel_db, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"  Ref Voice  : {args.ref_audio} ({len(audio_ref)/sr:.1f}s)")

    # 5. Synthesize
    print("\n  Generating speech audio...")
    audio_np, mel_np = synthesize(
        model, args.text, char2id, device,
        vocab_size=ckpt_vocab_size, ref_mel=ref_mel
    )

    # 6. Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, audio_np, cfg.sample_rate)
    duration = len(audio_np) / cfg.sample_rate

    print(f"\n  ✅ Speech Generated Successfully!")
    print(f"  Output File: {args.output}")
    print(f"  Duration   : {duration:.2f} seconds")
    print(f"  Sample Rate: {cfg.sample_rate} Hz")
    print(f"  Audio Shape: {audio_np.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
