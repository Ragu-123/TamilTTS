"""
TamilTTSv2 Preflight Gates — confirm the architecture is trainable BEFORE long runs.
====================================================================================
Gate 1  Vocoder Round-Trip   : real audio -> mel -> frozen HiFi-GAN -> audio must match
                                (validates the mel recipe vs the frozen vocoder).
Gate 2  Teacher-Forcing Fit  : model must drive mel loss toward zero on a tiny fixed set
                                using GT durations + GT f0/energy (isolates decoder/style
                                path from duration/prediction quality).
Gate 3  Gradient Flow Audit  : every trainable submodule must receive non-zero finite
                                gradients after one backward pass (catches dead branches,
                                e.g. an untrained aligner or pitch head).

Run:  python tests/preflight.py [--wav PATH] [--steps 300]
Exit code 0 = all gates passed.
"""
import argparse
import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from config import Config
from models import TamilTTSv2
from losses.losses import MelLoss, DurationLoss, PitchEnergyLoss


def make_tiny_cfg():
    """Full-size default model so param stats are realistic; gates stay cheap via small batches."""
    return Config()


def fake_batch(cfg, B=2, Tt=14, Tm=48, device="cpu"):
    Ta = Tm * cfg.hop_length
    tokens = torch.randint(6, cfg.vocab_size, (B, Tt), device=device)
    token_lens = torch.full((B,), Tt, dtype=torch.long, device=device)
    gt_dur = torch.zeros(B, Tt, device=device)
    per = max(1, Tm // Tt)
    rem = Tm - per * Tt
    gt_dur[:, :] = per
    gt_dur[:, -1] += rem
    audio = torch.randn(B, Ta, device=device) * 0.1
    log_f0 = (torch.randn(B, Tm, device=device))
    voiced = torch.ones(B, Tm, device=device)
    energy = torch.randn(B, Tm, device=device)
    return {
        "tokens": tokens, "token_lens": token_lens,
        "mel": torch.randn(B, 80, Tm, device=device) * 0.8 - 2.0,
        "mel_lens": torch.tensor([Tm] * B, device=device),
        "audio": audio, "audio_lens": torch.tensor([Ta] * B, device=device),
        "gt_dur": gt_dur, "log_f0": log_f0, "voiced": voiced, "energy": energy,
    }


def gate3_gradient_flow(model, batch, device):
    print("\n" + "=" * 60)
    print("GATE 3: Gradient Flow Audit (full regression objective)")
    print("=" * 60)
    ref_mel = torch.roll(batch["mel"], shifts=1, dims=0)      # exercise style_encoder
    out = model(
        batch["tokens"], batch["token_lens"],
        mel=batch["mel"], mel_lens=batch["mel_lens"],
        gt_dur=batch["gt_dur"], gt_logf0=batch["log_f0"], voiced=batch["voiced"],
        gt_energy=batch["energy"],
        ref_mel=ref_mel, style_dropout=0.5, return_audio=False,  # exercise default_style path too
    )
    mel_fn = MelLoss(coarse_w=0.5, refined_w=1.0)
    dur_fn = DurationLoss()
    pe_fn = PitchEnergyLoss()
    l_mel, _, _ = mel_fn(out["mel_pred"], out["mel_coarse"], batch["mel"], mel_lens=batch["mel_lens"])
    l_dur = dur_fn(out["log_dur"], batch["gt_dur"], token_lens=batch["token_lens"])
    l_f0, l_en = pe_fn(out["log_f0"], out["energy"], batch["log_f0"], batch["voiced"],
                       batch["energy"], mel_lens=batch["mel_lens"])
    loss = l_mel + l_dur + l_f0 + l_en
    loss.backward()

    bad = []
    checked = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        checked += 1
        g = p.grad
        if g is None or not torch.isfinite(g).all() or g.abs().sum().item() == 0.0:
            bad.append(name)

    top = sorted({n.split(".")[0] for n in [p[0] for p in model.named_parameters() if p[1].requires_grad]})
    print(f"  trainable params checked : {checked}")
    for t in top:
        sub = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad and n.startswith(t)]
        dead = sum(1 for _, g in sub if g is None or g.abs().sum().item() == 0.0)
        status = "OK " if dead == 0 else "DEAD"
        print(f"  [{status}] {t:<20s} ({len(sub)} tensors, {dead} zero-grad)")
    if bad:
        print(f"\n  FAIL: {len(bad)} parameter tensors received zero/non-finite gradients:")
        for n in bad[:20]:
            print(f"      {n}")
        return False
    print("  PASS: all trainable parameters receive non-zero finite gradients.")
    return True


def gate2_teacher_forcing_fit(model, cfg, device, steps=300):
    print("\n" + "=" * 60)
    print(f"GATE 2: Teacher-Forcing Fit ({steps} steps on 2 fixed batches)")
    print("=" * 60)
    batches = []
    for i in range(2):
        b = fake_batch(cfg, B=2, Tt=12 + i * 3, Tm=40 + i * 16, device=device)
        b["mel"] = torch.randn(2, 80, b["mel_lens"][0].item(), device=device) * 0.8 - 2.0
        batches.append(b)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    mel_fn = MelLoss(coarse_w=cfg.weight_mel_coarse, refined_w=cfg.weight_mel_refined)
    dur_fn = DurationLoss()
    pe_fn = PitchEnergyLoss()

    model.train()
    first = None
    last = None
    for step in range(steps):
        b = batches[step % len(batches)]
        out = model(
            b["tokens"], b["token_lens"], mel=b["mel"], mel_lens=b["mel_lens"],
            gt_dur=b["gt_dur"], gt_logf0=b["log_f0"], voiced=b["voiced"], gt_energy=b["energy"],
            ref_mel=torch.roll(b["mel"], 1, 0), ref_mel_lens=torch.roll(b["mel_lens"], 1),
            style_dropout=cfg.style_dropout_p, return_audio=False,
        )
        l_mel, _, _ = mel_fn(out["mel_pred"], out["mel_coarse"], b["mel"], mel_lens=b["mel_lens"])
        l_dur = dur_fn(out["log_dur"], b["gt_dur"], token_lens=b["token_lens"])
        l_f0, l_en = pe_fn(out["log_f0"], out["energy"], b["log_f0"], b["voiced"],
                           b["energy"], mel_lens=b["mel_lens"])
        loss = l_mel + cfg.weight_dur * l_dur + cfg.weight_f0 * l_f0 + cfg.weight_energy * l_en
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        v = l_mel.item()
        first = v if first is None else first
        last = v
        if step % max(steps // 10, 1) == 0 or step == steps - 1:
            print(f"  step {step:>5d}  mel_l1={v:.4f}")

    drop = (first - last) / max(first, 1e-9)
    ok = last < 0.35 and drop > 0.35
    print(f"  mel_l1 {first:.4f} -> {last:.4f}  (drop {drop*100:.0f}%)")
    print("  PASS: model fits fixed batches with GT prosody." if ok else "  FAIL: insufficient fit — check decoder/style/predictors wiring.")
    return ok


def gate1_vocoder_roundtrip(model, wav_path, device, out_path="preflight_roundtrip.wav"):
    print("\n" + "=" * 60)
    print("GATE 1: Vocoder Round-Trip (real audio -> mel -> HiFi-GAN)")
    print("=" * 60)
    import numpy as np
    import soundfile as sf

    if not wav_path or not os.path.exists(wav_path):
        print(f"  SKIP (no wav provided/found: {wav_path}). Pass --wav to enable.")
        return True

    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != cfg_sample_rate():
        import torchaudio.functional as AF
        audio = AF.resample(torch.from_numpy(audio).unsqueeze(0), sr, cfg_sample_rate()).squeeze(0).numpy()
    audio = torch.from_numpy(np.ascontiguousarray(audio)).to(device)

    from data.audio_features import MelExtractor
    mel_fn = MelExtractor(sample_rate=cfg_sample_rate(), n_fft=1024, hop_length=256,
                          n_mels=80, fmin=0.0, fmax=8000.0).to(device)
    mel = mel_fn(audio.unsqueeze(0)).transpose(1, 2)          # [1,T,80]

    model.eval()
    with torch.no_grad():
        rec = model.vocoder(mel)                               # [1,Ta]
    rec = rec.squeeze(0).cpu().numpy()
    sf.write(out_path, rec, cfg_sample_rate())

    min_len = min(len(audio), rec.shape[0])
    err = float(np.mean(np.abs(audio[:min_len].cpu().numpy() - rec[:min_len])))
    corr = float(np.corrcoef(audio[:min_len].cpu().numpy(), rec[:min_len])[0, 1])
    print(f"  wrote {out_path}")
    print(f"  waveform MAE={err:.4f}  corr={corr:.3f}  -> LISTEN to it: must sound identical.")
    ok = corr > 0.9
    print("  PASS." if ok else "  FAIL: correlation < 0.9 — mel recipe does NOT match frozen vocoder.")
    return ok


def cfg_sample_rate():
    return 22050


def find_default_wav(cfg):
    candidates = [
        r"C:\Users\SEC\Downloads\Tamil asr\ta_asr\download.wav",
    ]
    for d in getattr(cfg, "dataset_dir", []):
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".wav"):
                        candidates.append(os.path.join(root, f))
                break
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=str, default=None, help="real speech wav for round-trip gate")
    ap.add_argument("--steps", type=int, default=300, help="teacher-forcing fit steps")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = make_tiny_cfg()
    model = TamilTTSv2(cfg).to(device)

    total, train = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            train += p.numel()
    print("=" * 60)
    print(f"TamilTTSv2 preflight | trainable={train/1e6:.2f}M total={total/1e6:.2f}M (cap 80M)")
    print("=" * 60)
    budget_ok = total / 1e6 <= 80.0
    print("  PASS: parameter budget." if budget_ok else "  FAIL: exceeds 80M cap.")

    results = {"budget": budget_ok}
    results["grad_flow"] = gate3_gradient_flow(model, fake_batch(cfg, device=device), device)
    model.zero_grad(set_to_none=True)
    results["teacher_fit"] = gate2_teacher_forcing_fit(model, cfg, device, steps=args.steps)
    wav = args.wav or find_default_wav(cfg)
    results["round_trip"] = gate1_vocoder_roundtrip(model, wav, device)

    print("\n" + "=" * 60)
    print("PREFLIGHT SUMMARY")
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("=" * 60)
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
