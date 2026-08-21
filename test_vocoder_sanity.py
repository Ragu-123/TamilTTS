"""
HiFi-GAN Vocoder Sanity Verification Script
===========================================
Tests the pre-trained Frozen HiFi-GAN Vocoder independently by passing a real audio WAV file
through 22.05 kHz Mel extraction and direct HiFi-GAN resynthesis.

If the output audio sounds crisp and natural, it confirms 100% that:
1. The vocoder checkpoint is valid.
2. The 22.05 kHz mel spectrogram parameters (n_fft=1024, hop=256, fmin=0, fmax=8000, log clamp [-11.5, 0.0])
   are perfectly matched to the vocoder filters.

Usage:
    python test_vocoder_sanity.py --audio /path/to/reference.wav --vocoder_ckpt /path/to/hifigan_generator.pt --output ./sanity_output.wav
"""
import argparse
import os
import torch
import numpy as np
import soundfile as sf
import librosa
from models.vocoder import load_pretrained_vocoder


def compute_mel_22k(audio_path, target_sr=22050, n_fft=1024, hop_length=256, n_mels=80, fmin=0.0, fmax=8000.0):
    audio, sr = librosa.load(audio_path, sr=target_sr)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=target_sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels,
        fmin=fmin, fmax=fmax,
    )
    mel_log = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    mel_log = np.clip(mel_log, a_min=-11.5, a_max=0.0)
    return mel_log, audio, sr


def main():
    parser = argparse.ArgumentParser(description="HiFi-GAN Ground-Truth Mel Sanity Test")
    parser.add_argument("--audio", type=str, required=True, help="Input real WAV audio file")
    parser.add_argument("--vocoder_ckpt", type=str, default="/kaggle/input/notebooks/sanjaynn/tamiltts-vocoder/indic_tts_tamil_clean/hifigan_generator.pt",
                        help="Path to HiFi-GAN generator.pt")
    parser.add_argument("--output", type=str, default="sanity_reconstructed.wav", help="Output reconstructed WAV file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  HiFi-GAN Ground-Truth Mel Re-Synthesis Sanity Test")
    print("=" * 60)
    print(f"  Input Audio : {args.audio}")
    print(f"  Vocoder Ckpt: {args.vocoder_ckpt}")
    print(f"  Device      : {device}")

    # 1. Load audio and compute 22.05 kHz Mel
    mel_log, orig_audio, sr = compute_mel_22k(args.audio, target_sr=22050)
    mel_tensor = torch.tensor(mel_log, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 80, T_mel]
    print(f"  Mel Shape   : {mel_tensor.shape} (Frames: {mel_tensor.shape[2]}, Duration: {len(orig_audio)/sr:.2f}s)")

    # 2. Load Frozen Vocoder
    vocoder = load_pretrained_vocoder(device=device, checkpoint_path=args.vocoder_ckpt)
    vocoder.eval()

    # 3. Direct Mel -> Audio Synthesis
    with torch.no_grad():
        reconstructed_audio = vocoder(mel_tensor)  # [1, T_audio]

    audio_np = reconstructed_audio.squeeze(0).cpu().numpy()

    # Normalize
    audio_np = audio_np - np.mean(audio_np)
    max_val = np.abs(audio_np).max()
    if max_val > 0.01:
        audio_np = (audio_np / max_val) * 0.95

    # 4. Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    sf.write(args.output, audio_np, 22050)

    print("\n  ✅ Sanity Test Re-Synthesis Complete!")
    print(f"  Output File : {args.output}")
    print(f"  Sample Rate : 22050 Hz")
    print(f"  Duration    : {len(audio_np)/22050:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
