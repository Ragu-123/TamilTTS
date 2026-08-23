"""
Overfit quality test: train a small set of real Tamil clips until synthesized
audio is intelligible. Use this to find the step count that gives good audio.

Validated recipe (from Kaggle iteration experiments):
  - style_dropout = 0.0 (dropout raised the mel floor and blurred mels)
  - warmup + cosine decay (constant LR caused loss spikes)
  - honest FULL-set eval logging (per-batch prints were misleading)
  - reference conditioning from a DIFFERENT clip (matches training distribution)

Usage:
  python overfit_quality.py --steps 8000
  python overfit_quality.py --steps 12000 --n_samples 20

Outputs -> /kaggle/working/ttsv2_overfit_quality/
  model_overfit.pt, s{i}_original.wav, s{i}_teacherforced.wav, s{i}_freerun.wav, texts.txt
"""
import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Subset

from config import Config
from data import build_tamil_datasets
from data.dataset import tamil_tts_collate_fn
from losses.losses import DurationLoss, MelLoss, PitchEnergyLoss
from models import TamilTTSv2
from preprocess.g2g import TAMIL_G2G_TOKENS


def decode_tokens(ids):
    return " ".join(
        TAMIL_G2G_TOKENS[int(t)] if 0 <= int(t) < len(TAMIL_G2G_TOKENS) else "?"
        for t in ids
    )


def pick_clips(train_ds, n_samples):
    picks = []
    for i in range(0, min(len(train_ds), 20000), 25):
        try:
            item = train_ds[i]
        except Exception:
            continue
        if item is None:
            continue
        if 2.0 < float(item[5]) / cfg.sample_rate < 8.0:
            picks.append(i)
        if len(picks) >= n_samples:
            break
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--n_samples", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="/kaggle/working/ttsv2_overfit_quality")
    args = ap.parse_args()

    global cfg
    cfg = Config()
    device = "cuda"
    outd = args.out
    os.makedirs(outd, exist_ok=True)
    torch.manual_seed(1234)
    t0 = time.time()

    print("[1] loading dataset...", flush=True)
    train_ds, _ = build_tamil_datasets(cfg.dataset_dir, cfg)
    picks = pick_clips(train_ds, args.n_samples)
    items = [train_ds[p] for p in picks]
    batches = [
        b
        for b in DataLoader(
            Subset(train_ds, picks), batch_size=args.batch_size, shuffle=False,
            collate_fn=tamil_tts_collate_fn,
        )
        if b is not None
    ]
    print(f"[1] {len(items)} clips ({sum(1 for _ in batches)} batches) ({time.time()-t0:.0f}s)", flush=True)

    model = TamilTTSv2(cfg).to(device)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    mel_fn, dur_fn, pe_fn = MelLoss(0.5, 1.0), DurationLoss(), PitchEnergyLoss()

    TOTAL, WARM = args.steps, min(500, max(100, args.steps // 20))
    PEAK = args.lr

    def lr_at(s):
        if s < WARM:
            return PEAK * (s + 1) / WARM
        p_ = (s - WARM) / max(1, TOTAL - WARM)
        return 3e-6 + 0.5 * (PEAK - 3e-6) * (1 + math.cos(math.pi * p_))

    @torch.no_grad()
    def full_eval():
        was_training = model.training
        model.eval()
        vals = []
        for b in batches:
            t = b["tokens"].to(device); tl = b["token_lens"].to(device)
            m = b["mel"].to(device); ml = b["mel_lens"].to(device)
            gd = b["gt_dur"].to(device); f0 = b["log_f0"].to(device)
            vc = b["voiced"].to(device); en = b["energy"].to(device)
            o = model(t, tl, mel=m, mel_lens=ml, gt_dur=gd, gt_logf0=f0,
                      voiced=vc, gt_energy=en,
                      ref_mel=torch.roll(m, 1, dims=0), ref_mel_lens=torch.roll(ml, 1),
                      style_dropout=0.0, return_audio=False)
            l1, _, _ = mel_fn(o["mel_pred"], o["mel_coarse"], m, mel_lens=ml)
            vals.append(l1.item())
        if was_training:
            model.train()
        return sum(vals) / len(vals)

    print(f"[2] training {TOTAL} steps (warmup {WARM}) ...", flush=True)
    rng = random.Random(7)
    order = list(range(len(batches)))
    log_every = max(250, TOTAL // 12)
    for step in range(1, TOTAL + 1):
        if (step - 1) % len(batches) == 0:
            rng.shuffle(order)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = batches[order[(step - 1) % len(batches)]]
        t = b["tokens"].to(device); tl = b["token_lens"].to(device)
        m = b["mel"].to(device); ml = b["mel_lens"].to(device)
        gd = b["gt_dur"].to(device); f0 = b["log_f0"].to(device)
        vc = b["voiced"].to(device); en = b["energy"].to(device)
        o = model(t, tl, mel=m, mel_lens=ml, gt_dur=gd, gt_logf0=f0,
                  voiced=vc, gt_energy=en,
                  ref_mel=torch.roll(m, 1, dims=0), ref_mel_lens=torch.roll(ml, 1),
                  style_dropout=0.0, return_audio=False)
        l1, _, _ = mel_fn(o["mel_pred"], o["mel_coarse"], m, mel_lens=ml)
        ld = dur_fn(o["log_dur"], gd, token_lens=tl)
        lf, le = pe_fn(o["log_f0"], o["energy"], f0, vc, en, mel_lens=ml)
        loss = l1 + ld + lf + le
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 0.5)
        opt.step()
        if step % log_every == 0 or step == TOTAL:
            print(f"[2] step {step} | FULL mel {full_eval():.4f} | "
                  f"lr {lr_at(step):.2e} | {time.time()-t0:.0f}s", flush=True)

    torch.save({"model_state_dict": model.state_dict()}, f"{outd}/model_overfit.pt")
    print("[3] checkpoint saved", flush=True)

    # ---- synthesize first 3 clips: original / teacher-forced / free-run ----
    def pearson(a, b):
        return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])

    model.eval()
    lines = []
    for idx in range(min(3, len(items))):
        tok_i, tlen_i, mel_i, mlen_i, audio_i, alen_i, gtd_i, *_ = items[idx]
        T = min(mlen_i, int(gtd_i.sum()))
        j = 3 if idx != 3 else 5
        ref = items[j][2].unsqueeze(0)
        rl = torch.tensor([items[j][3]])
        tt = torch.tensor(tok_i).unsqueeze(0).to(device)
        tll = torch.tensor([tlen_i]).to(device)
        kw = dict(ref_mel=ref.to(device), ref_mel_lens=rl.to(device),
                  style_dropout=0.0, return_audio=True)
        with torch.no_grad():
            otf = model(tt, tll, mel=mel_i.unsqueeze(0).to(device),
                        mel_lens=torch.tensor([mlen_i]).to(device),
                        gt_dur=torch.tensor(gtd_i, dtype=torch.float32).unsqueeze(0).to(device), **kw)
            ofr = model(tt, tll, mel_lens=torch.tensor([mlen_i]).to(device), **kw)
        p = otf["mel_pred"][0, :T, :].cpu().numpy()
        tg = mel_i[:, :T].numpy().T
        l1 = float(np.abs(p - tg).mean())
        c = pearson(p, tg)
        sf.write(f"{outd}/s{idx}_original.wav", audio_i.numpy(), cfg.sample_rate)
        sf.write(f"{outd}/s{idx}_teacherforced.wav", otf["gen_audio"][0].cpu().numpy(), cfg.sample_rate)
        sf.write(f"{outd}/s{idx}_freerun.wav", ofr["gen_audio"][0].cpu().numpy(), cfg.sample_rate)
        lines.append(f"s{idx}\tL1={l1:.4f}\tcorr={c:.3f}\tGT: {decode_tokens(tok_i.tolist())}")
        print(lines[-1], flush=True)

    with open(f"{outd}/texts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[4] wavs written to {outd}", flush=True)
    print(f"QUALITY BAR: listen to s0_teacherforced.wav vs s0_original.wav; "
          f"if still robotic rerun with --steps {(TOTAL * 3) // 2}", flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
