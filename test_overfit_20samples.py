"""
TamilTTS 20-Sample Overfit & Rapid Convergence Test
===================================================
Quickly trains on 20 fixed ground-truth MFA samples for 500-1000 steps.
Verifies that:
1. Duration Loss drops towards ~0.01 (Duration predictor memorizes ground-truth MFA timing).
2. Mel L1 Loss drops towards ~0.10 (Acoustic model reconstructs natural 22.05kHz speech).
3. Synthesizes .wav files to verify clean, natural pacing and absence of robotic buzzing.
"""
import os
import time
import glob
import torch
import numpy as np
import soundfile as sf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config.config import Config
from data.dataset import DirectParquetTamilDataset, tamil_tts_collate_fn
from models.tamil_tts import TamilTTS
from losses.losses import DualMelLoss, LogDurationLoss
from utils.utils import count_parameters


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 65)
    print("  TamilTTS 20-Sample Rapid Overfit Verification")
    print(f"  Device: {device}")
    print("=" * 65)

    # 1. Parquet files
    parquet_files = []
    for d in cfg.dataset_dir:
        found = sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True))
        parquet_files.extend(found)

    if not parquet_files:
        print("❌ Error: No parquet files found.")
        return

    # 2. Dataset strictly loaded with 100% MFA ground-truth durations
    full_dataset = DirectParquetTamilDataset(parquet_files, cfg)
    num_overfit = min(20, len(full_dataset))
    overfit_indices = list(range(num_overfit))
    dataset = Subset(full_dataset, overfit_indices)
    print(f"  ✓ Subsampled {len(dataset)} fixed samples for rapid overfit test.\n")

    loader = DataLoader(
        dataset,
        batch_size=min(4, num_overfit),
        shuffle=True,
        collate_fn=tamil_tts_collate_fn,
        num_workers=0
    )

    # 3. Model
    model = TamilTTS(cfg).to(device)
    if cfg.vocoder_ckpt and os.path.exists(cfg.vocoder_ckpt):
        from models.vocoder import load_pretrained_vocoder
        voc = load_pretrained_vocoder(device=device, checkpoint_path=cfg.vocoder_ckpt)
        model.vocoder.load_state_dict(voc.state_dict(), strict=False)
        for p in model.vocoder.parameters():
            p.requires_grad = False
        print(f"  ✓ Frozen Vocoder loaded from {cfg.vocoder_ckpt}")

    total_p, train_p = count_parameters(model)
    print(f"  Trainable Parameters: {train_p / 1e6:.2f}M\n")

    # 4. Optimizer & Loss Functions (Optimized for Rapid Overfit Convergence)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2000, eta_min=1e-4)
    dual_mel_fn = DualMelLoss(coarse_weight=cfg.weight_mel_coarse, refined_weight=cfg.weight_mel_refined)
    dur_loss_fn = LogDurationLoss()

    output_dir = "./overfit_audio_outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 5. Overfit Loop (2,000 steps for deep convergence)
    total_steps = 2000
    step = 0
    start_time = time.time()
    pbar = tqdm(total=total_steps, desc="Overfitting 20 Samples", ncols=110)

    while step < total_steps:
        for batch in loader:
            if batch is None:
                continue

            text_tokens, text_lens, ref_mel, mel_lens, real_audio, audio_lens, batch_gt_dur = batch
            text_tokens  = text_tokens.to(device)
            text_lens    = text_lens.to(device)
            ref_mel      = ref_mel.to(device)
            mel_lens     = mel_lens.to(device)
            real_audio   = real_audio.to(device)
            target_dur   = batch_gt_dur.to(device).float()

            optimizer.zero_grad()

            gen_audio, mel_refined, mel_coarse, dur_pred, log_dur_pred, align_dur, _, _, _ = model(
                text_tokens, text_lens,
                ref_mel=ref_mel,
                mel_lens=mel_lens,
                target_dur=target_dur,
                return_audio=False
            )

            # Mel loss
            mel_target = ref_mel.transpose(1, 2)
            loss_mel, _, _ = dual_mel_fn(mel_refined, mel_coarse, mel_target, mel_lens=mel_lens)

            # Duration loss
            loss_dur = dur_loss_fn(log_dur_pred, target_dur, text_lens=text_lens)

            total_loss = 45.0 * loss_mel + cfg.weight_dur * loss_dur
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            step += 1
            pbar.update(1)
            pbar.set_postfix({
                "mel_loss": f"{loss_mel.item():.4f}",
                "dur_loss": f"{loss_dur.item():.4f}",
                "total": f"{total_loss.item():.2f}"
            })

            if step >= total_steps:
                break

    pbar.close()
    elapsed = time.time() - start_time
    print(f"\n🎉 2000-step overfit finished in {elapsed:.1f}s!")

    # 6. Dual-Mode Evaluation & Audio Synthesis
    print("\n" + "=" * 65)
    print("  Synthesizing Test Audio (Evaluating Acoustic Model & Duration Predictor)")
    print("=" * 65)
    model.eval()
    with torch.no_grad():
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            toks, t_len, sample_mel, m_len, _, _, true_dur = sample
            toks_t = toks.unsqueeze(0).to(device)
            t_len_t = torch.tensor([t_len], dtype=torch.long, device=device)
            sample_mel_t = sample_mel.unsqueeze(0).to(device)
            true_dur_t = true_dur.unsqueeze(0).to(device).float()

            # Test A: Pure Acoustic Network Synthesis (Using Original Ground-Truth MFA Durations + Speaker Style)
            audio_mfa, mel_mfa, _, _, _, _, _, _, _ = model(
                toks_t, t_len_t,
                ref_mel=sample_mel_t,
                target_dur=true_dur_t,
                return_audio=True
            )
            wav_mfa = audio_mfa.squeeze().cpu().numpy()
            max_val = np.abs(wav_mfa).max()
            if max_val > 0.001:
                wav_mfa = (wav_mfa / max_val) * 0.95
            mfa_path = os.path.join(output_dir, f"sample_{i+1}_mfa_durations.wav")
            sf.write(mfa_path, wav_mfa, cfg.sample_rate)

            # Test B: Duration Predictor Inference Mode (Using Model-Predicted Durations)
            audio_pred, mel_pred, _, dur_pred, _, _, _, _, _ = model(
                toks_t, t_len_t,
                ref_mel=sample_mel_t,
                return_audio=True
            )
            wav_pred = audio_pred.squeeze().cpu().numpy()
            max_val_pred = np.abs(wav_pred).max()
            if max_val_pred > 0.001:
                wav_pred = (wav_pred / max_val_pred) * 0.95
            pred_path = os.path.join(output_dir, f"sample_{i+1}_predicted_durations.wav")
            sf.write(pred_path, wav_pred, cfg.sample_rate)

            dur_mfa_sec = len(wav_mfa) / cfg.sample_rate
            dur_pred_sec = len(wav_pred) / cfg.sample_rate
            print(f"\n[Sample {i+1}] Graphemes: {t_len}")
            print(f"  🔊 Test A (MFA Durations)      : {dur_mfa_sec:.2f}s audio -> {mfa_path}")
            print(f"  🤖 Test B (Predicted Durations): {dur_pred_sec:.2f}s audio -> {pred_path}")

    print(f"\n✅ All sample WAVs saved to '{output_dir}'.")


if __name__ == "__main__":
    main()
