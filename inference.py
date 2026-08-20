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


def build_char2id():
    """Build the exact same character vocabulary used during training."""
    char2id = {" ": 1}  # 0=PAD, 1=SPACE
    idx = 2
    for c in range(0x0B80, 0x0C00):
        char2id[chr(c)] = idx
        idx += 1
    for d in "0123456789":
        char2id[d] = idx
        idx += 1
    for p in list(".,!?;:-'\""):
        char2id[p] = idx
        idx += 1
    return char2id


def text_to_ids(text, char2id, max_text_len=200):
    """Convert Tamil text to token IDs (identical to training tokenization)."""
    text = normalize_tamil_text(text)
    ids = [char2id.get(ch, 0) for ch in text]
    ids = ids[:max_text_len]
    ids += [0] * (max_text_len - len(ids))
    return ids


@torch.no_grad()
def synthesize(model, text, char2id, device, ref_mel=None, max_text_len=200):
    """
    Generate speech audio from Tamil text.

    Args:
        model:    Loaded TamilTTS model (eval mode)
        text:     Tamil text string
        char2id:  Character-to-ID mapping
        device:   torch device
        ref_mel:  Optional reference mel [1, 80, T] for voice cloning.
                  If None, uses a neutral zero-style vector.
    Returns:
        audio_np: numpy array of generated audio at 16kHz
        mel_np:   numpy array of predicted mel spectrogram
    """
    # 1. Tokenize text
    token_ids = text_to_ids(text, char2id, max_text_len)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)  # [1, T_text]

    # 2. Create reference mel (for style extraction)
    if ref_mel is None:
        # Use a short silence mel as neutral style reference
        ref_mel = torch.zeros(1, 80, 50, device=device)  # [1, 80, 50]
    elif ref_mel.dim() == 2:
        ref_mel = ref_mel.unsqueeze(0)  # [80, T] -> [1, 80, T]
    ref_mel = ref_mel.to(device)

    # 3. Forward pass (inference — no target_mel_len, model uses duration predictor freely)
    audio, mel_pred, dur_pred = model(tokens, ref_mel, target_mel_len=None)

    # 4. Convert to numpy
    audio_np = audio.squeeze(0).cpu().numpy()  # [T_audio]
    mel_np = mel_pred.squeeze(0).cpu().numpy()  # [T_mel, 80]

    # 5. Remove trailing silence (trim padding)
    # Find last non-silent frame from duration predictions
    dur_np = dur_pred.squeeze(0).cpu().numpy()
    non_pad = (np.array(token_ids) != 0)
    total_dur_frames = int(np.sum(np.round(np.clip(dur_np[non_pad], 0, None))))
    if total_dur_frames > 0:
        # Trim audio to predicted speech length
        hop_length = 256
        trim_samples = min(total_dur_frames * hop_length, len(audio_np))
        audio_np = audio_np[:trim_samples]

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

    # 1. Load model
    print("=" * 60)
    print("  TamilTTS Inference")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Text       : {args.text}")

    model = TamilTTS(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded from step {ckpt['step']}, loss {ckpt['loss']:.4f}")

    # 2. Build vocabulary
    char2id = build_char2id()

    # 3. Load reference audio for voice cloning (optional)
    ref_mel = None
    if args.ref_audio:
        import librosa
        audio_ref, sr = librosa.load(args.ref_audio, sr=cfg.sample_rate)
        mel = librosa.feature.melspectrogram(
            y=audio_ref, sr=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, n_mels=cfg.mel_channels,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        ref_mel = torch.tensor(mel_db, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"  Ref Audio  : {args.ref_audio} ({len(audio_ref)/sr:.1f}s)")

    # 4. Synthesize
    print("\n  Generating speech...")
    audio_np, mel_np = synthesize(model, args.text, char2id, device, ref_mel=ref_mel)

    # 5. Save output
    sf.write(args.output, audio_np, cfg.sample_rate)
    duration = len(audio_np) / cfg.sample_rate
    print(f"\n  ✅ Audio saved: {args.output}")
    print(f"  Duration   : {duration:.2f}s")
    print(f"  Sample Rate: {cfg.sample_rate} Hz")
    print(f"  Mel Frames : {mel_np.shape[0]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
