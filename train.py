"""
TamilTTS Training Script
========================
- Multi-GPU via nn.DataParallel (auto-detects GPU count)
- tqdm progress bars
- Checkpointing every N steps
- Resume from checkpoint
- Cosine LR with warmup
- 300,000 steps for Stage 1
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from lion_pytorch import Lion
from transformers import WhisperModel, WhisperFeatureExtractor, WavLMModel

from config import Config
from models import TamilTTS
from losses import SLMLoss, SRFDLoss
from data import get_dataloader
from utils import save_checkpoint, load_checkpoint, count_parameters, get_lr_scheduler


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ==================== MODEL ====================
    print("=" * 60)
    print("  TamilTTS Training Pipeline")
    print("=" * 60)

    model = TamilTTS(cfg).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"  Total Parameters   : {total_params / 1e6:.2f}M")
    print(f"  Trainable Parameters: {trainable_params / 1e6:.2f}M")

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"  GPUs Detected      : {gpu_count} (DataParallel enabled)")
        model = nn.DataParallel(model)
    else:
        print(f"  GPUs Detected      : {gpu_count}")

    # ==================== OPTIMIZER & SCHEDULER ====================
    optimizer = Lion(model.parameters(), lr=cfg.learning_rate)
    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps)

    # ==================== CRITICS ====================
    print(f"  Loading WavLM from : {cfg.wavlm_dir}")
    wavlm = WavLMModel.from_pretrained(cfg.wavlm_dir).to(device)
    slm_loss_fn = SLMLoss(wavlm)

    print(f"  Loading IndicWhisper: {cfg.whisper_dir}")
    whisper_enc = WhisperModel.from_pretrained(cfg.whisper_dir).encoder.to(device)
    whisper_ext = WhisperFeatureExtractor.from_pretrained(cfg.whisper_dir)
    srfd_loss_fn = SRFDLoss(whisper_enc, whisper_ext)

    # ==================== DATASET ====================
    print(f"  Dataset            : {cfg.dataset_dir}")
    dataloader = get_dataloader(cfg)
    steps_per_epoch = len(dataloader)
    total_epochs = (cfg.total_steps // steps_per_epoch) + 1
    print(f"  Steps per epoch    : {steps_per_epoch}")
    print(f"  Total steps target : {cfg.total_steps}")
    print(f"  Epochs needed      : {total_epochs}")
    print("=" * 60)

    # ==================== RESUME ====================
    global_step = 0
    resume_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
    if os.path.exists(resume_path):
        global_step = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"  Resuming from step {global_step}")

    # ==================== TRAINING LOOP ====================
    model.train()
    best_loss = float("inf")

    for epoch in range(total_epochs):
        if global_step >= cfg.total_steps:
            break

        pbar = tqdm(
            enumerate(dataloader),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{total_epochs}",
            unit="step",
            ncols=120,
        )

        for step, (text_tokens, ref_mel, real_audio) in pbar:
            if global_step >= cfg.total_steps:
                break

            text_tokens = text_tokens.to(device)
            ref_mel = ref_mel.to(device)
            real_audio = real_audio.to(device)

            # --- Forward ---
            optimizer.zero_grad()
            gen_audio, mel_pred, dur_pred = model(text_tokens, ref_mel)

            # --- Losses ---
            # 1. Mel L1 (basic phonetic alignment)
            # Use the real mel extracted from dataset for target
            mel_target = ref_mel  # [B, 80, T_mel]
            min_mel_len = min(mel_pred.size(1), mel_target.size(2))
            loss_mel = F.l1_loss(
                mel_pred[:, :min_mel_len, :],
                mel_target[:, :, :min_mel_len].transpose(1, 2)
            )

            # 2. SLM Loss (WavLM adversarial)
            min_audio_len = min(gen_audio.size(1), real_audio.size(1))
            loss_slm = slm_loss_fn(
                real_audio[:, :min_audio_len],
                gen_audio[:, :min_audio_len],
            )

            # 3. SR-FD Loss (IndicWhisper phonetic)
            loss_srfd = srfd_loss_fn(
                real_audio[:, :min_audio_len],
                gen_audio[:, :min_audio_len],
            )

            # 4. Total
            total_loss = (
                cfg.weight_mel * loss_mel
                + cfg.weight_slm * loss_slm
                + cfg.weight_srfd * loss_srfd
            )

            # --- Backward ---
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1

            # --- Logging ---
            pbar.set_postfix({
                "step": global_step,
                "loss": f"{total_loss.item():.4f}",
                "mel": f"{loss_mel.item():.4f}",
                "slm": f"{loss_slm.item():.4f}",
                "srfd": f"{loss_srfd.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            # --- Checkpoint ---
            if global_step % cfg.save_every == 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step, total_loss.item(),
                    os.path.join(cfg.checkpoint_dir, "latest.pt"),
                )
                # Save numbered checkpoint too
                save_checkpoint(
                    model, optimizer, scheduler, global_step, total_loss.item(),
                    os.path.join(cfg.checkpoint_dir, f"step_{global_step}.pt"),
                )

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                save_checkpoint(
                    model, optimizer, scheduler, global_step, best_loss,
                    os.path.join(cfg.checkpoint_dir, "best.pt"),
                )

    # --- Final Save ---
    save_checkpoint(
        model, optimizer, scheduler, global_step, total_loss.item(),
        os.path.join(cfg.checkpoint_dir, "final.pt"),
    )
    print("Training Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TamilTTS Training")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--wavlm_dir", type=str, default=None)
    parser.add_argument("--whisper_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    cfg = Config()
    # Override config with CLI args if provided
    if args.dataset_dir: cfg.dataset_dir = args.dataset_dir
    if args.wavlm_dir: cfg.wavlm_dir = args.wavlm_dir
    if args.whisper_dir: cfg.whisper_dir = args.whisper_dir
    if args.checkpoint_dir: cfg.checkpoint_dir = args.checkpoint_dir
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.total_steps: cfg.total_steps = args.total_steps

    if args.resume:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        import shutil
        shutil.copy(args.resume, os.path.join(cfg.checkpoint_dir, "latest.pt"))

    train(cfg)
