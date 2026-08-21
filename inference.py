"""
TamilTTS Inference Script — v5 (SOTA AI4Bharat / Kokoro Architecture)
=====================================================================
Generates natural, expressive Tamil speech audio from text using trained TamilTTS checkpoint.

Features:
- Natural duration pacing directly from learned DurationPredictor (no artificial frame clamping).
- Reference voice extraction for zero-shot voice cloning and natural expressive intonation.
- PostNet 5-layer convolutional formant refinement.
- Pre-trained Frozen Universal HiFi-GAN Vocoder integration.
- Speed scaling (--speed) and pitch/style conditioning (--ref_audio).

Usage:
    # 1. Voice Cloning (Recommended: use any clean Tamil WAV audio as voice reference)
    python inference.py --text "வணக்கம், நான் தமிழில் பேசுகிறேன்." --checkpoint ./checkpoints/best.pt --ref_audio sample.wav

    # 2. Default Synthesis
    python inference.py --text "வணக்கம், நான் தமிழில் பேசுகிறேன்." --checkpoint ./checkpoints/best.pt
"""
import argparse
import os
import glob
import re
import torch
import numpy as np
import soundfile as sf
import librosa
from config import Config
from models import TamilTTS
from models.vocoder import load_pretrained_vocoder
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


def build_char2id(max_vocab_size=256):
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
    # Digits 0-9
    for d in "0123456789":
        if idx < max_vocab_size:
            char2id[d] = idx
            idx += 1
    # Punctuation
    for p in list(".,!?;:-'\"()"):
        if idx < max_vocab_size:
            char2id[p] = idx
            idx += 1
    return char2id


def text_to_ids(text, char2id, max_text_len=200, max_vocab_size=256):
    """Convert Tamil text to token IDs safely bounded by vocab_size."""
    text = normalize_tamil_text(text)
    ids = []
    for ch in text:
        token_id = char2id.get(ch, 0)
        if token_id >= max_vocab_size:
            token_id = 0
        ids.append(token_id)

    ids = ids[:max_text_len]
    ids += [0] * (max_text_len - len(ids))
    return ids


def compute_mel_from_audio(audio_path, cfg):
    """Computes clamped log-mel spectrogram matching Kokoro/HiFi-GAN scale."""
    audio_ref, sr = librosa.load(audio_path, sr=cfg.sample_rate)
    mel = librosa.feature.melspectrogram(
        y=audio_ref, sr=cfg.sample_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, n_mels=cfg.mel_channels,
        fmin=0.0, fmax=8000.0,
    )
    mel_log = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    mel_log = np.clip(mel_log, a_min=-11.5, a_max=0.0)
    return mel_log, len(audio_ref) / sr


@torch.no_grad()
def synthesize(model, text, char2id, device, vocab_size=256, ref_mel=None,
               max_text_len=200, speed=1.0, external_vocoder=None):
    """
    Generate speech audio from Tamil text using learned durations and acoustic projections.
    """
    # 1. Tokenize text with safe vocab clamping
    token_ids = text_to_ids(text, char2id, max_text_len, max_vocab_size=vocab_size)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)  # [1, T_text]
    text_mask = (tokens == 0)                                            # [1, T_text]

    # 2. Reference mel for style extraction (Natural Log-Mel scale)
    if ref_mel is None:
        # Generate a standard acoustic reference frame distribution
        ref_mel = torch.full((1, 80, 50), -3.5, device=device)
    elif ref_mel.dim() == 2:
        ref_mel = ref_mel.unsqueeze(0)
    ref_mel = ref_mel.to(device)

    # 3. Model forward pass
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()

    # Step A: Text encoding with Sinusoidal Positional Encoding & pad mask
    style = eval_model.style_encoder(ref_mel)
    x = eval_model.text_encoder(tokens, mask=text_mask)

    # Step B: Duration prediction directly from learned predictor
    dur_pred, log_dur_pred = eval_model.duration_predictor(x, mask=text_mask)  # [1, T_text]

    # Apply user speed scaling (1.0 = normal, 0.8 = slower, 1.2 = faster)
    non_pad = (~text_mask).float()
    dur_scaled = dur_pred * (1.0 / max(speed, 0.1))
    dur_scaled = torch.clamp(dur_scaled, min=1.0) * non_pad  # Minimum 1 frame per character

    total_frames = int(torch.round(dur_scaled).sum().item())
    total_frames = max(total_frames, 16)

    # Forward through model using regulated durations
    audio, mel_refined, mel_coarse, _, _ = eval_model(
        tokens, ref_mel,
        target_mel_len=total_frames,
        text_mask=text_mask,
        target_dur=dur_scaled
    )

    # If external universal vocoder is provided, use it for waveform synthesis
    if external_vocoder is not None:
        audio = external_vocoder(mel_refined)

    # 4. Post-process audio waveform
    audio_np = audio.squeeze(0).cpu().numpy()
    mel_np = mel_refined.squeeze(0).cpu().numpy()

    # Remove DC offset
    audio_np = audio_np - np.mean(audio_np)

    # Peak normalization
    max_val = np.abs(audio_np).max()
    if max_val > 0.01:
        audio_np = (audio_np / max_val) * 0.95

    return audio_np, mel_np


def main():
    parser = argparse.ArgumentParser(description="TamilTTS Inference Pipeline (v5 — SOTA Architecture)")
    parser.add_argument("--text", type=str, required=True, help="Tamil text to synthesize")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (best.pt)")
    parser.add_argument("--output", type=str, default="output_tamil.wav", help="Output WAV file path")
    parser.add_argument("--ref_audio", type=str, default=None, help="Optional: reference audio WAV for voice cloning / speaker pitch")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (default: 1.0, 0.85=slower/clearer)")
    parser.add_argument("--vocoder_ckpt", type=str, default=None,
                        help="Path to pre-trained universal HiFi-GAN vocoder checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    print("=" * 60)
    print("  TamilTTS Inference Pipeline (v5 — SOTA Architecture)")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Text       : {args.text}")
    print(f"  Speed      : {args.speed}x")

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
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"  Loaded from: Step {ckpt.get('step', 0)}, Loss {ckpt.get('loss', 0.0):.4f}")

    # 3. Build vocabulary bounded by checkpoint vocab size
    char2id = build_char2id(max_vocab_size=ckpt_vocab_size)

    # 4. Optional voice cloning reference audio
    ref_mel = None
    if args.ref_audio and os.path.exists(args.ref_audio):
        mel_log, ref_dur = compute_mel_from_audio(args.ref_audio, cfg)
        ref_mel = torch.tensor(mel_log, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"  Ref Voice  : {args.ref_audio} ({ref_dur:.1f}s)")
    else:
        # Check if any sample audio exists in dataset for natural default voice
        sample_audio_candidates = glob.glob("/kaggle/input/datasets/**/*.wav", recursive=True) or glob.glob("./*.wav")
        if sample_audio_candidates and os.path.exists(sample_audio_candidates[0]):
            sample_file = sample_audio_candidates[0]
            mel_log, ref_dur = compute_mel_from_audio(sample_file, cfg)
            ref_mel = torch.tensor(mel_log, dtype=torch.float32, device=device).unsqueeze(0)
            print(f"  Default Ref: Using dataset voice {os.path.basename(sample_file)} ({ref_dur:.1f}s)")

    # 5. Load pre-trained Universal Vocoder
    vocoder_path = args.vocoder_ckpt or cfg.vocoder_ckpt
    external_vocoder = load_pretrained_vocoder(device=device, checkpoint_path=vocoder_path)

    # 6. Synthesize
    print("\n  Generating speech audio...")
    audio_np, mel_np = synthesize(
        model, args.text, char2id, device,
        vocab_size=ckpt_vocab_size,
        ref_mel=ref_mel,
        speed=args.speed,
        external_vocoder=external_vocoder,
    )

    # 7. Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
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
