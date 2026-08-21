"""
TamilTTS Inference Script — v6 (Natural Speech Pacing & Voice Cloning)
======================================================================
Generates natural, human-paced Tamil speech audio using trained checkpoint and voice reference.

Features:
- Natural Speech Pacing (auto-scales early checkpoints to natural 80-110ms per Tamil syllable).
- Zero-Shot Voice Cloning via Reference Audio (--ref_audio).
- PostNet 5-layer harmonic formant refinement.
- Pre-trained Frozen Universal HiFi-GAN Vocoder (22.05 kHz output).
"""
import argparse
import os
import re
import torch
import numpy as np
import soundfile as sf
import librosa
from config import Config
from models import TamilTTS
from models.vocoder import load_pretrained_vocoder
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

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
    char2id = {" ": 1}
    idx = 2
    for c in range(0x0B80, 0x0C00):
        if idx < max_vocab_size:
            char2id[chr(c)] = idx
            idx += 1
    for d in "0123456789":
        if idx < max_vocab_size:
            char2id[d] = idx
            idx += 1
    for p in list(".,!?;:-'\"()"):
        if idx < max_vocab_size:
            char2id[p] = idx
            idx += 1
    return char2id


def text_to_ids(text, char2id, max_text_len=200, max_vocab_size=256):
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


def compute_mel_from_audio(audio_path, target_sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    """Computes clamped log-mel spectrogram matching Kokoro/HiFi-GAN scale."""
    audio_ref, sr = librosa.load(audio_path, sr=target_sr)
    mel = librosa.feature.melspectrogram(
        y=audio_ref, sr=target_sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
        fmin=0.0, fmax=8000.0,
    )
    mel_log = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    mel_log = np.clip(mel_log, a_min=-11.5, a_max=0.0)
    return mel_log, len(audio_ref) / sr


@torch.no_grad()
def synthesize(model, text, char2id, device, vocab_size=256, ref_mel=None,
               max_text_len=200, speed=1.0, frames_per_char=6.0,
               external_vocoder=None):
    """
    Synthesize natural-speed Tamil speech audio.
    
    frames_per_char: Target frames per Tamil character (default: 6.0 frames = ~96ms per letter).
                     For a 35-char sentence, this produces a natural ~3.3 second utterance.
    """
    token_ids = text_to_ids(text, char2id, max_text_len, max_vocab_size=vocab_size)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
    text_mask = (tokens == 0)
    non_pad = (~text_mask).float()
    num_chars = int(non_pad.sum().item())

    # Reference mel for voice style
    if ref_mel is None:
        ref_mel = torch.full((1, 80, 100), -3.5, device=device)
    elif ref_mel.dim() == 2:
        ref_mel = ref_mel.unsqueeze(0)
    ref_mel = ref_mel.to(device)

    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()

    # Step A: Text encoding with Positional Encodings
    style = eval_model.style_encoder(ref_mel)
    x = eval_model.text_encoder(tokens, mask=text_mask)

    # Step B: Duration prediction & Human Speech Pacing
    dur_pred, _ = eval_model.duration_predictor(x, mask=text_mask)

    # Check if duration predictor has learned full scaling (early checkpoints predict ~1 frame)
    avg_pred = (dur_pred * non_pad).sum() / max(num_chars, 1)
    if avg_pred.item() < 3.5:
        # Scale to natural human Tamil speech cadence (~6.0 frames per character)
        pace_scale = (frames_per_char / max(avg_pred.item(), 0.5)) * (1.0 / max(speed, 0.1))
        dur_scaled = dur_pred * pace_scale
    else:
        dur_scaled = dur_pred * (1.0 / max(speed, 0.1))

    # Ensure minimum 3 frames for vowels/consonants and zero for pad
    dur_scaled = torch.clamp(dur_scaled, min=3.0) * non_pad

    total_frames = int(torch.round(dur_scaled).sum().item())
    total_frames = max(total_frames, 32)

    # Forward through model using regulated durations
    audio, mel_refined, mel_coarse, _, _, _, _ = eval_model(
        tokens, ref_mel,
        target_mel_len=total_frames,
        text_mask=text_mask,
        target_dur=dur_scaled
    )

    # Vocoder waveform synthesis
    if external_vocoder is not None:
        audio = external_vocoder(mel_refined)

    audio_np = audio.squeeze(0).cpu().numpy()
    mel_np = mel_refined.squeeze(0).cpu().numpy()

    # Post-process
    audio_np = audio_np - np.mean(audio_np)
    max_val = np.abs(audio_np).max()
    if max_val > 0.01:
        audio_np = (audio_np / max_val) * 0.95

    return audio_np, mel_np


def main():
    parser = argparse.ArgumentParser(description="TamilTTS Inference Pipeline (v6 — Natural Speech Pacing)")
    parser.add_argument("--text", type=str, required=True, help="Tamil text to synthesize")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (best.pt)")
    parser.add_argument("--output", type=str, default="output_tamil.wav", help="Output WAV file path")
    parser.add_argument("--ref_audio", type=str, default=None, help="Reference audio WAV for speaker voice cloning")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (1.0 = normal, 0.85 = slower/clearer)")
    parser.add_argument("--pace", type=float, default=6.0,
                        help="Target mel frames per Tamil character (default: 6.0 = ~96ms per letter / natural human pace)")
    parser.add_argument("--vocoder_ckpt", type=str, default=None,
                        help="Path to pre-trained universal HiFi-GAN vocoder checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    print("=" * 60)
    print("  TamilTTS Inference Pipeline (v6 — Natural Speech Pacing)")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Text       : {args.text}")
    print(f"  Speed      : {args.speed}x | Target Pace: {args.pace} frames/char")

    # 1. Match vocab size from checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    emb_weight = ckpt["model_state_dict"].get("text_encoder.embedding.weight")
    if emb_weight is not None:
        ckpt_vocab_size = emb_weight.shape[0]
        cfg.vocab_size = ckpt_vocab_size
        print(f"  Vocab Size : {ckpt_vocab_size}")
    else:
        ckpt_vocab_size = cfg.vocab_size

    # 2. Instantiate and load model
    model = TamilTTS(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"  Loaded from: Step {ckpt.get('step', 0)}, Loss {ckpt.get('loss', 0.0):.4f}")

    char2id = build_char2id(max_vocab_size=ckpt_vocab_size)

    # 3. Load speaker reference voice
    ref_mel = None
    if args.ref_audio and os.path.exists(args.ref_audio):
        mel_log, ref_dur = compute_mel_from_audio(args.ref_audio, target_sr=cfg.sample_rate)
        ref_mel = torch.tensor(mel_log, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"  Voice Ref  : {args.ref_audio} ({ref_dur:.1f}s)")
    else:
        print("  Voice Ref  : Default Natural Acoustic Distribution")

    # 4. Load Vocoder
    vocoder_path = args.vocoder_ckpt or cfg.vocoder_ckpt
    external_vocoder = load_pretrained_vocoder(device=device, checkpoint_path=vocoder_path)

    # 5. Synthesize
    print("\n  Generating natural-paced speech audio...")
    audio_np, mel_np = synthesize(
        model, args.text, char2id, device,
        vocab_size=ckpt_vocab_size,
        ref_mel=ref_mel,
        speed=args.speed,
        frames_per_char=args.pace,
        external_vocoder=external_vocoder,
    )

    # 6. Save Audio
    # HiFi-GAN V1 outputs audio at 22,050 Hz (256x upsampling of 80 mel frames)
    # or 16,000 Hz depending on native training sample rate
    output_sr = 22050 if cfg.sample_rate == 22050 or "indic_tts" in str(vocoder_path) else cfg.sample_rate
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    sf.write(args.output, audio_np, output_sr)
    duration = len(audio_np) / output_sr

    print(f"\n  ✅ Speech Generated Successfully!")
    print(f"  Output File: {args.output}")
    print(f"  Duration   : {duration:.2f} seconds (Natural Human Speed)")
    print(f"  Sample Rate: {output_sr} Hz")
    print(f"  Audio Shape: {audio_np.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
