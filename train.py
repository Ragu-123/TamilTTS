"""
TamilTTS Training Script
========================
- 300,000 steps for Stage 1 (390 hours Shrutilipi)
- 95/5 train/val split
- Multi-GPU via nn.DataParallel
- tqdm progress bars
- Checkpoint every 5000 steps + best model
- Resume from checkpoint
- Cosine LR with 10k warmup
- Validation with SR-FD metric via IndicWhisper
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
from data import build_dataloaders
from utils import save_checkpoint, load_checkpoint, count_parameters, get_lr_scheduler


# ==================== VALIDATION ====================
@torch.no_grad()
def validate(model, val_loader, srfd_loss_fn, device, global_step):
    """Run validation: compute mel L1 + SR-FD metric."""
    model.eval()
    total_mel_loss = 0.0
    total_srfd     = 0.0
    num_batches    = 0
    max_val_batches = 50  # Cap validation to keep it fast

    pbar = tqdm(val_loader, desc="  Validating", ncols=100, leave=False)
    for i, (text_tokens, ref_mel, real_audio) in enumerate(pbar):
        if i >= max_val_batches:
            break

        text_tokens = text_tokens.to(device)
        ref_mel     = ref_mel.to(device)
        real_audio  = real_audio.to(device)

        gen_audio, mel_pred, dur_pred = model(text_tokens, ref_mel)

        # Mel L1 (compare predicted mel against extracted mel)
        mel_target = ref_mel.transpose(1, 2)  # [B, 80, T] -> [B, T, 80]
        min_t = min(mel_pred.size(1), mel_target.size(1))
        mel_loss = F.l1_loss(mel_pred[:, :min_t, :], mel_target[:, :min_t, :])
        total_mel_loss += mel_loss.item()

        # SR-FD (IndicWhisper phonetic quality metric)
        min_a = min(gen_audio.size(1), real_audio.size(1))
        srfd = srfd_loss_fn(real_audio[:, :min_a], gen_audio[:, :min_a])
        total_srfd += srfd.item()

        num_batches += 1
        pbar.set_postfix(mel=f"{mel_loss.item():.4f}", srfd=f"{srfd.item():.4f}")

    avg_mel  = total_mel_loss / max(num_batches, 1)
    avg_srfd = total_srfd / max(num_batches, 1)
    model.train()
    return avg_mel, avg_srfd


# ==================== MAIN TRAINING ====================
def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Banner ---
    print("=" * 60)
    print("  TamilTTS Training Pipeline")
    print("=" * 60)

    # --- Model ---
    model = TamilTTS(cfg).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"  Total Parameters    : {total_params / 1e6:.2f}M")
    print(f"  Trainable Parameters: {trainable_params / 1e6:.2f}M")

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"  GPUs Detected       : {gpu_count} (DataParallel enabled)")
        model = nn.DataParallel(model)
    else:
        print(f"  GPUs Detected       : {gpu_count}")

    # --- Optimizer & Scheduler ---
    optimizer = Lion(model.parameters(), lr=cfg.learning_rate)
    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps)

    # --- Critics ---
    print(f"  Loading WavLM       : {cfg.wavlm_dir}")
    wavlm = WavLMModel.from_pretrained(cfg.wavlm_dir).to(device)
    slm_loss_fn = SLMLoss(wavlm)

    print(f"  Loading IndicWhisper: {cfg.whisper_dir}")
    whisper_enc = WhisperModel.from_pretrained(cfg.whisper_dir).encoder.to(device)
    whisper_ext = WhisperFeatureExtractor.from_pretrained(cfg.whisper_dir)
    srfd_loss_fn = SRFDLoss(whisper_enc, whisper_ext)

    # --- Dataset ---
    print(f"  Dataset             : {cfg.dataset_dir}")
    train_loader, val_loader = build_dataloaders(cfg)
    steps_per_epoch = len(train_loader)
    total_epochs = (cfg.total_steps // steps_per_epoch) + 1
    print(f"  Train steps/epoch   : {steps_per_epoch}")
    print(f"  Val batches         : {len(val_loader)}")
    print(f"  Total steps target  : {cfg.total_steps}")
    print(f"  Epochs needed       : {total_epochs}")
    print("=" * 60)

    # --- Resume ---
    global_step = 0
    best_val_loss = float("inf")
    resume_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
    if os.path.exists(resume_path):
        global_step = load_checkpoint(resume_path, model, optimizer, scheduler)

    # --- Training Loop ---
    model.train()
    for epoch in range(total_epochs):
        if global_step >= cfg.total_steps:
            break

        pbar = tqdm(
            enumerate(train_loader), total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{total_epochs}", unit="step", ncols=130,
        )

        for step, (text_tokens, ref_mel, real_audio) in pbar:
            if global_step >= cfg.total_steps:
                break

            text_tokens = text_tokens.to(device)
            ref_mel     = ref_mel.to(device)
            real_audio  = real_audio.to(device)

            # --- Forward ---
            optimizer.zero_grad()
            gen_audio, mel_pred, dur_pred = model(text_tokens, ref_mel)

            # --- Loss 1: Mel L1 ---
            mel_target = ref_mel.transpose(1, 2)  # [B, 80, T] -> [B, T, 80]
            min_mel_t = min(mel_pred.size(1), mel_target.size(1))
            loss_mel = F.l1_loss(mel_pred[:, :min_mel_t, :], mel_target[:, :min_mel_t, :])

            # --- Loss 2: SLM (WavLM adversarial) ---
            min_audio_t = min(gen_audio.size(1), real_audio.size(1))
            loss_slm = slm_loss_fn(real_audio[:, :min_audio_t], gen_audio[:, :min_audio_t])

            # --- Loss 3: Duration Huber ---
            # Target: approximate duration from audio/text ratio
            dur_target = torch.full_like(dur_pred, fill_value=real_audio.size(1) / dur_pred.size(1))
            loss_dur = F.huber_loss(dur_pred, dur_target)

            # --- Total Loss ---
            total_loss = (
                cfg.weight_mel * loss_mel
                + cfg.weight_slm * loss_slm
                + cfg.weight_dur * loss_dur
            )

            # --- Backward ---
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            # --- Progress Bar ---
            pbar.set_postfix({
                "step": global_step,
                "loss": f"{total_loss.item():.3f}",
                "mel": f"{loss_mel.item():.3f}",
                "slm": f"{loss_slm.item():.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}",
            })

            # --- Checkpoint ---
            if global_step % cfg.save_every == 0:
                save_checkpoint(model, optimizer, scheduler, global_step, total_loss.item(),
                                os.path.join(cfg.checkpoint_dir, "latest.pt"))
                save_checkpoint(model, optimizer, scheduler, global_step, total_loss.item(),
                                os.path.join(cfg.checkpoint_dir, f"step_{global_step}.pt"))
                print(f"\n  [Checkpoint] Saved at step {global_step}")

            # --- Validation ---
            if global_step % cfg.val_every == 0:
                val_mel, val_srfd = validate(model, val_loader, srfd_loss_fn, device, global_step)
                print(f"\n  [Val @ step {global_step}] Mel L1: {val_mel:.4f} | SR-FD: {val_srfd:.4f}")

                if val_mel < best_val_loss:
                    best_val_loss = val_mel
                    save_checkpoint(model, optimizer, scheduler, global_step, val_mel,
                                    os.path.join(cfg.checkpoint_dir, "best.pt"))
                    print(f"  [Val] New best model saved! (Mel L1: {val_mel:.4f})")

    # --- Final Save ---
    save_checkpoint(model, optimizer, scheduler, global_step, total_loss.item(),
                    os.path.join(cfg.checkpoint_dir, "final.pt"))
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Final step: {global_step}")
    print(f"  Best val Mel L1: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TamilTTS Training")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--wavlm_dir", type=str, default=None)
    parser.add_argument("--whisper_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_dir:    cfg.dataset_dir = args.dataset_dir
    if args.wavlm_dir:      cfg.wavlm_dir = args.wavlm_dir
    if args.whisper_dir:     cfg.whisper_dir = args.whisper_dir
    if args.checkpoint_dir:  cfg.checkpoint_dir = args.checkpoint_dir
    if args.batch_size:      cfg.batch_size = args.batch_size
    if args.total_steps:     cfg.total_steps = args.total_steps

    if args.resume:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        import shutil
        shutil.copy(args.resume, os.path.join(cfg.checkpoint_dir, "latest.pt"))

    train(cfg)
