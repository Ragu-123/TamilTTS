"""
TamilTTS Training Script — v4 (DDP)
=====================================
- DistributedDataParallel: All 4 GPUs at ~95-100% utilization
- Gradient Accumulation: Effective batch = per_gpu_batch * num_gpus * accum_steps
- OOM Prevention: Small per-GPU batch (8) + accumulation (4) = effective batch 128
- Each GPU has its own WavLM + IndicWhisper (no bottleneck on GPU 0)
- Fallback: Works on single GPU without torchrun
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from lion_pytorch import Lion
from transformers import WhisperModel, WhisperFeatureExtractor, WavLMModel

from config import Config
from models import TamilTTS
from losses import SLMLoss, SRFDLoss
from data import ShrutilipiDataset
from utils import save_checkpoint, load_checkpoint, count_parameters, get_lr_scheduler
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_main_process():
    """Only rank 0 prints, saves checkpoints, and runs validation."""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    """Initialize DDP process group. Returns local_rank and device."""
    if "LOCAL_RANK" not in os.environ:
        # Not launched with torchrun → single GPU fallback
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, device, False

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, device, True


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def log(msg):
    """Print only on rank 0."""
    if is_main_process():
        print(msg)


# ---------------------------------------------------------------------------
# Validation (rank 0 only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, srfd_loss_fn, device, global_step):
    was_training = model.training
    model.eval()
    # Use the unwrapped model for validation (avoid DDP forward hooks)
    eval_model = model.module if hasattr(model, "module") else model

    total_mel = 0.0
    total_srfd = 0.0
    n = 0
    max_val = 50

    pbar = tqdm(val_loader, desc="  Validating", ncols=100, leave=False)
    for i, (text_tokens, ref_mel, real_audio) in enumerate(pbar):
        if i >= max_val:
            break
        text_tokens = text_tokens.to(device)
        ref_mel     = ref_mel.to(device)
        real_audio  = real_audio.to(device)

        target_mel_len = ref_mel.size(2)
        gen_audio, mel_pred, _ = eval_model(text_tokens, ref_mel, target_mel_len=target_mel_len)

        mel_target = ref_mel.transpose(1, 2)
        mel_loss = F.l1_loss(mel_pred, mel_target)
        total_mel += mel_loss.item()

        min_a = min(gen_audio.size(1), real_audio.size(1))
        srfd = srfd_loss_fn(real_audio[:, :min_a], gen_audio[:, :min_a])
        total_srfd += srfd.item()

        n += 1
        pbar.set_postfix(mel=f"{mel_loss.item():.4f}", srfd=f"{srfd.item():.4f}")

    if was_training:
        model.train()
    return total_mel / max(n, 1), total_srfd / max(n, 1)


# ---------------------------------------------------------------------------
# Build DataLoaders (with DistributedSampler for DDP)
# ---------------------------------------------------------------------------

def build_ddp_dataloaders(cfg, is_distributed):
    """Build train/val DataLoaders with DistributedSampler when using DDP."""
    log("Loading Shrutilipi dataset...")
    full_ds = load_dataset("parquet", data_dir=cfg.dataset_dir, split="train")
    split = full_ds.train_test_split(test_size=cfg.val_split, seed=42)

    log(f"  Train samples: {len(split['train'])}")
    log(f"  Val samples  : {len(split['test'])}")

    train_ds = ShrutilipiDataset(
        split["train"], max_audio_len=cfg.max_audio_len, max_text_len=cfg.max_text_len,
        mel_channels=cfg.mel_channels, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )
    val_ds = ShrutilipiDataset(
        split["test"], max_audio_len=cfg.max_audio_len, max_text_len=cfg.max_text_len,
        mel_channels=cfg.mel_channels, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )

    # DDP: each GPU sees a unique shard of the data
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_distributed else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False)  if is_distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def train(cfg):
    local_rank, device, is_distributed = setup_ddp()
    world_size = dist.get_world_size() if is_distributed else 1

    log("=" * 60)
    log("  TamilTTS Training Pipeline (v4 — DDP + Gradient Accumulation)")
    log("=" * 60)

    # --- Model ---
    model = TamilTTS(cfg).to(device)
    total_p, train_p = count_parameters(model)
    log(f"  Total Parameters    : {total_p / 1e6:.2f}M")
    log(f"  Trainable Parameters: {train_p / 1e6:.2f}M")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        log(f"  DDP Active          : {world_size} GPUs (each sees {cfg.per_gpu_batch} samples)")
    else:
        gpu_count = torch.cuda.device_count()
        log(f"  Single GPU Mode     : {gpu_count} GPU(s) detected")

    effective_batch = cfg.per_gpu_batch * world_size * cfg.grad_accum_steps
    log(f"  Per-GPU Batch       : {cfg.per_gpu_batch}")
    log(f"  Gradient Accum Steps: {cfg.grad_accum_steps}")
    log(f"  Effective Batch Size: {effective_batch}")

    # --- Optimizer ---
    optimizer = Lion(model.parameters(), lr=cfg.learning_rate)
    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps)

    # --- Critics (each GPU gets its own copy — no bottleneck!) ---
    log(f"  Loading WavLM       : {cfg.wavlm_dir}")
    wavlm = WavLMModel.from_pretrained(cfg.wavlm_dir).to(device)
    slm_loss_fn = SLMLoss(wavlm)

    log(f"  Loading IndicWhisper: {cfg.whisper_dir}")
    whisper_enc = WhisperModel.from_pretrained(cfg.whisper_dir).encoder.to(device)
    whisper_ext = WhisperFeatureExtractor.from_pretrained(cfg.whisper_dir)
    srfd_loss_fn = SRFDLoss(whisper_enc, whisper_ext)

    # --- Data ---
    log(f"  Dataset             : {cfg.dataset_dir}")
    train_loader, val_loader, train_sampler = build_ddp_dataloaders(cfg, is_distributed)
    steps_per_epoch = len(train_loader)
    total_epochs = (cfg.total_steps // steps_per_epoch) + 1
    log(f"  Train steps/epoch   : {steps_per_epoch}")
    log(f"  Val batches         : {len(val_loader)}")
    log(f"  Total steps target  : {cfg.total_steps}")
    log(f"  Epochs needed       : {total_epochs}")
    log("=" * 60)

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

        # DDP: reshuffle data each epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(
            enumerate(train_loader), total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{total_epochs}", unit="step", ncols=140,
            disable=not is_main_process(),
        )

        optimizer.zero_grad()

        for step, (text_tokens, ref_mel, real_audio) in pbar:
            if global_step >= cfg.total_steps:
                break

            text_tokens = text_tokens.to(device)
            ref_mel     = ref_mel.to(device)
            real_audio  = real_audio.to(device)
            target_mel_len = ref_mel.size(2)

            # Forward
            gen_audio, mel_pred, dur_pred = model(
                text_tokens, ref_mel, target_mel_len=target_mel_len
            )

            # --- Losses ---
            mel_target = ref_mel.transpose(1, 2)
            loss_mel = F.l1_loss(mel_pred, mel_target)

            min_a = min(gen_audio.size(1), real_audio.size(1))
            loss_slm = slm_loss_fn(real_audio[:, :min_a], gen_audio[:, :min_a])

            non_pad = (text_tokens != 0).float()
            target_dur = non_pad * (target_mel_len / non_pad.sum(dim=1, keepdim=True).clamp(min=1))
            loss_dur = F.huber_loss(dur_pred, target_dur)

            total_loss = (
                cfg.weight_mel * loss_mel
                + cfg.weight_slm * loss_slm
                + cfg.weight_dur * loss_dur
            )

            # Scale loss by accumulation steps (so gradients average correctly)
            scaled_loss = total_loss / cfg.grad_accum_steps
            scaled_loss.backward()

            # --- Gradient Accumulation: only step every N micro-batches ---
            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main_process():
                    pbar.set_postfix({
                        "step": global_step,
                        "loss": f"{total_loss.item():.3f}",
                        "mel": f"{loss_mel.item():.3f}",
                        "slm": f"{loss_slm.item():.3f}",
                        "dur": f"{loss_dur.item():.3f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                    })

                # --- Checkpoint (rank 0 only) ---
                if is_main_process() and global_step % cfg.save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, total_loss.item(),
                                    os.path.join(cfg.checkpoint_dir, "latest.pt"))
                    save_checkpoint(model, optimizer, scheduler, global_step, total_loss.item(),
                                    os.path.join(cfg.checkpoint_dir, f"step_{global_step}.pt"))
                    print(f"\n  [Checkpoint] Saved at step {global_step}")

                # --- Validation (rank 0 only) ---
                if is_main_process() and global_step % cfg.val_every == 0:
                    val_mel, val_srfd = validate(model, val_loader, srfd_loss_fn, device, global_step)
                    print(f"\n  [Val @ step {global_step}] Mel L1: {val_mel:.4f} | SR-FD: {val_srfd:.4f}")
                    if val_mel < best_val_loss:
                        best_val_loss = val_mel
                        save_checkpoint(model, optimizer, scheduler, global_step, val_mel,
                                        os.path.join(cfg.checkpoint_dir, "best.pt"))
                        print(f"  [Val] New best model saved! (Mel L1: {val_mel:.4f})")

                # Synchronize all GPUs after validation/checkpoint
                if is_distributed:
                    dist.barrier()

    # --- Final Save ---
    if is_main_process():
        save_checkpoint(model, optimizer, scheduler, global_step, total_loss.item(),
                        os.path.join(cfg.checkpoint_dir, "final.pt"))
        print("\n" + "=" * 60)
        print("  Training Complete!")
        print(f"  Final step: {global_step}")
        print(f"  Best val Mel L1: {best_val_loss:.4f}")
        print("=" * 60)

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TamilTTS DDP Training")
    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--wavlm_dir", type=str, default=None)
    parser.add_argument("--whisper_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--per_gpu_batch", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_dir:      cfg.dataset_dir = args.dataset_dir
    if args.wavlm_dir:        cfg.wavlm_dir = args.wavlm_dir
    if args.whisper_dir:      cfg.whisper_dir = args.whisper_dir
    if args.checkpoint_dir:   cfg.checkpoint_dir = args.checkpoint_dir
    if args.per_gpu_batch:    cfg.per_gpu_batch = args.per_gpu_batch
    if args.grad_accum_steps: cfg.grad_accum_steps = args.grad_accum_steps
    if args.total_steps:      cfg.total_steps = args.total_steps

    if args.resume:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        import shutil
        shutil.copy(args.resume, os.path.join(cfg.checkpoint_dir, "latest.pt"))

    train(cfg)
