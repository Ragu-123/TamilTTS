"""
TamilTTSv2 Command-Line Inference
=================================
Loads a checkpoint (preferring EMA weights), tokenizes Tamil text with the G2G tokenizer,
optionally conditions on a reference audio's style, and writes a 22.05 kHz waveform.
"""
import argparse

import numpy as np
import soundfile as sf
import torch

from config import Config
from data.audio_features import MelExtractor
from models.tamil_tts_v2 import TamilTTSv2
from preprocess.g2g import TAMIL_G2G_TOKENS, segment_tamil_g2g


def build_encoder():
    tok2id = {tok: i for i, tok in enumerate(TAMIL_G2G_TOKENS)}
    unk_id = tok2id.get("<unk>", 2)

    def encode(text, max_len):
        segmented = segment_tamil_g2g(text, set(TAMIL_G2G_TOKENS))
        ids = [tok2id.get(tok, unk_id) for tok in segmented.split()]
        ids = ids[:max_len] or [unk_id]
        tokens = torch.tensor([ids], dtype=torch.long)
        token_lens = torch.tensor([len(ids)], dtype=torch.long)
        return tokens, token_lens

    return encode


def load_reference_mel(ref_path, cfg, device):
    import torchaudio.functional as AF
    wav, sr = sf.read(ref_path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(np.ascontiguousarray(wav))
    if sr != cfg.sample_rate:
        wav_t = AF.resample(wav_t, orig_freq=sr, new_freq=cfg.sample_rate).float()
    extractor = MelExtractor(
        sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.mel_channels, fmin=cfg.f_min, fmax=cfg.f_max,
    ).to(device).eval()
    with torch.no_grad():
        mel = extractor(wav_t.to(device))
    ref_mel = mel.unsqueeze(0)
    ref_mel_lens = torch.tensor([mel.size(1)], dtype=torch.long, device=device)
    return ref_mel, ref_mel_lens


def load_model(cfg, checkpoint_path, device):
    model = TamilTTSv2(cfg).to(device).eval()
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob.get("ema_state_dict") or blob["model_state_dict"]
        step = blob.get("step", "?")
    else:
        state, step = blob, "?"
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  ⚠️ Missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  ⚠️ Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    print(f"  ✅ Loaded checkpoint {checkpoint_path} (step {step}, EMA={isinstance(blob, dict) and bool(blob.get('ema_state_dict'))})")
    return model


def main():
    parser = argparse.ArgumentParser(description="TamilTTSv2 inference")
    parser.add_argument("--text", required=True, help="Tamil text to synthesize")
    parser.add_argument("--checkpoint", default="./checkpoints/best.pt", help="Checkpoint path (.pt)")
    parser.add_argument("--out", default="output.wav", help="Output wav path")
    parser.add_argument("--ref_audio", default=None, help="Optional reference wav for style conditioning")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or e.g. 'cuda:0'")
    args = parser.parse_args()

    cfg = Config()
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    encode = build_encoder()
    tokens, token_lens = encode(args.text, cfg.max_text_len)
    tokens, token_lens = tokens.to(device), token_lens.to(device)

    model = load_model(cfg, args.checkpoint, device)

    ref_mel = ref_mel_lens = None
    if args.ref_audio:
        ref_mel, ref_mel_lens = load_reference_mel(args.ref_audio, cfg, device)
        print(f"  🎙️ Style reference: {args.ref_audio}")

    with torch.no_grad():
        out = model(
            tokens, token_lens,
            gt_dur=None,
            ref_mel=ref_mel, ref_mel_lens=ref_mel_lens,
            return_audio=True,
        )

    gen_audio = out.get("gen_audio") if isinstance(out, dict) else None
    if gen_audio is None:
        raise RuntimeError("Model did not return audio (gen_audio is None).")

    wav = gen_audio[0].float().cpu().numpy().astype(np.float32)
    sf.write(args.out, wav, cfg.sample_rate)
    print(f"✅ Synthesized {args.out} ({wav.shape[-1] / cfg.sample_rate:.2f}s @ {cfg.sample_rate} Hz)")


if __name__ == "__main__":
    main()
