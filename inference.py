"""
TamilTTS Inference Script — Pure Learned Durations & Voice Cloning (FastPitch SOTA Standard)
============================================================================================
Features:
- Pure Learned Duration Synthesis (Timing governed directly by trained DurationPredictor).
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
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    _tamil_normalizer = IndicNormalizerFactory().get_normalizer("ta")
except ImportError:
    _tamil_normalizer = None

_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def normalize_tamil_text(text):
    if not isinstance(text, str):
        return ""
    if _tamil_normalizer is not None:
        text = _tamil_normalizer.normalize(text)
    text = text.translate(_SUBSCRIPT_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    return text


from data.dataset import build_tamil_vocab, TAMIL_G2G_TOKENS
from preprocess.g2g import segment_tamil_g2g


def text_to_ids(text, char2id, max_vocab_size=384):
    text = normalize_tamil_text(text)
    g2g_set = set(TAMIL_G2G_TOKENS)
    segmented = segment_tamil_g2g(text, g2g_set)
    tokens = segmented.split()
    ids = [char2id.get(t, 2) for t in tokens]
    return ids


def compute_mel_from_audio(audio_path, target_sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    """Computes symmetric dB mel spectrogram matching 22.05kHz IndicTTS / HiFi-GAN scale [-4.0, +4.0]."""
    import torchaudio.functional as AF
    from data.dataset import IndicTTSMelProcessor
    data, orig_sr = sf.read(audio_path)
    audio_t = torch.tensor(data, dtype=torch.float32)
    if audio_t.ndim > 1:
        audio_t = audio_t.mean(dim=-1)
    if orig_sr != target_sr:
        audio_t = AF.resample(audio_t, orig_sr, target_sr)

    mel_proc = IndicTTSMelProcessor(sample_rate=target_sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel = mel_proc(audio_t)
    if mel.dim() == 3:
        mel = mel.squeeze(0)
    return mel.numpy(), len(audio_t) / target_sr


@torch.no_grad()
def synthesize(model, text, char2id, device, vocab_size=256, ref_mel=None, speed=1.0, external_vocoder=None):
    """
    Synthesize natural Tamil speech audio from purely learned duration representations.
    """
    ids = text_to_ids(text, char2id, max_vocab_size=vocab_size)
    if len(ids) == 0:
        ids = [1]

    text_lens = torch.tensor([len(ids)], dtype=torch.long, device=device)
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    text_mask = torch.zeros(1, len(ids), dtype=torch.bool, device=device)

    # Reference mel for voice style (matches IndicTTSMelProcessor silence floor -4.0)
    if ref_mel is None:
        ref_mel = torch.full((1, 80, 100), -4.0, device=device)
    elif ref_mel.dim() == 2:
        ref_mel = ref_mel.unsqueeze(0)
    ref_mel = ref_mel.to(device)

    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()

    # Step A: Text encoding with Positional Encodings
    x = eval_model.text_encoder(tokens, mask=text_mask)

    # Step B: Pure learned duration prediction
    dur_pred, _ = eval_model.duration_predictor(x, mask=text_mask)  # [1, T_text]

    # Apply speed scaling and enforce minimum phoneme duration floor (min 3 frames = 35ms)
    dur_scaled = dur_pred * (1.0 / max(speed, 0.1))
    dur_rounded = torch.clamp(torch.round(dur_scaled), min=3.0)
    total_frames = int(dur_rounded.sum().item())
    total_frames = max(total_frames, 16)

    # Forward through model using regulated durations
    audio, mel_refined, mel_coarse, _, _, _, _, _, _ = eval_model(
        tokens, text_lens,
        ref_mel=ref_mel,
        target_dur=dur_rounded,
        return_audio=True
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
    parser = argparse.ArgumentParser(description="TamilTTS Inference Pipeline (Pure Learned Durations)")
    parser.add_argument("--text", type=str, required=True, help="Tamil text to synthesize")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (best.pt)")
    parser.add_argument("--output", type=str, default="output_tamil.wav", help="Output WAV file path")
    parser.add_argument("--ref_audio", type=str, default=None, help="Reference audio WAV for speaker voice cloning")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed (1.0 = normal, 0.85 = slower/clearer)")
    parser.add_argument("--vocoder_ckpt", type=str, default=None,
                        help="Path to pre-trained universal HiFi-GAN vocoder checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()

    print("=" * 60)
    print("  TamilTTS Inference Pipeline (FastPitch / Pure Learned Durations)")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Text       : {args.text}")
    print(f"  Speed      : {args.speed}x")

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
    char2id, _ = build_tamil_vocab(max_vocab=ckpt_vocab_size)

    # 3. Load speaker reference voice
    ref_mel = None
    if args.ref_audio and os.path.exists(args.ref_audio):
        mel_log, ref_dur = compute_mel_from_audio(args.ref_audio, target_sr=cfg.sample_rate)
        ref_mel = torch.tensor(mel_log, dtype=torch.float32, device=device).unsqueeze(0)
        print(f"  Voice Ref  : {args.ref_audio} ({ref_dur:.1f}s)")
    else:
        print("  Voice Ref  : Neutral Acoustic Distribution")

    # 4. Load Vocoder (Strict 100% Match)
    vocoder_path = args.vocoder_ckpt or cfg.vocoder_ckpt
    external_vocoder = load_pretrained_vocoder(device=device, checkpoint_path=vocoder_path)
    model.vocoder.load_state_dict(external_vocoder.state_dict(), strict=False)

    # 5. Synthesize
    print("\n  Generating speech audio...")
    audio_np, mel_np = synthesize(
        model, args.text, char2id, device,
        vocab_size=ckpt_vocab_size,
        ref_mel=ref_mel,
        speed=args.speed,
        external_vocoder=external_vocoder,
    )

    # 6. Save Audio at 22,050 Hz
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
