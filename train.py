"""
TamilTTS Training Script — v5 (Auto-DDP Multi-GPU)
===================================================
- Auto-Detects 4 GPUs: Works with BOTH `python train.py` and `torchrun`!
- DistributedDataParallel: All 4 GPUs at ~95-100% compute evenly.
- Per-GPU Critic Instances: Each GPU has its own WavLM + IndicWhisper.
- OOM Prevention: per_gpu_batch=8, grad_accum_steps=4 (Effective batch = 128).
- Progress & Metrics: Clear Epoch, Global Optimizer Step, and Loss Tracking.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
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

def is_main_process(rank=0):
    return rank == 0


def log(msg, rank=0):
    if is_main_process(rank):
        print(msg)


# ---------------------------------------------------------------------------
# Validation (Rank 0 only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, val_loader, srfd_loss_fn, device, global_step):
    was_training = model.training
    model.eval()
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
# DDP Worker Function
# ---------------------------------------------------------------------------

def train_worker(local_rank, world_size, cfg):
    # 1. Setup Process Group
    is_distributed = world_size > 1
    if is_distributed:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=local_rank,
                world_size=world_size
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 60, local_rank)
    log("  TamilTTS Training Pipeline (v5 — Auto-DDP Multi-GPU)", local_rank)
    log("=" * 60, local_rank)

    # 2. Model
    model = TamilTTS(cfg).to(device)
    total_p, train_p = count_parameters(model)
    log(f"  Total Parameters    : {total_p / 1e6:.2f}M", local_rank)
    log(f"  Trainable Parameters: {train_p / 1e6:.2f}M", local_rank)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        log(f"  🔥 DDP Active       : {world_size} GPUs (Rank {local_rank} on cuda:{local_rank})", local_rank)
    else:
        log(f"  Single GPU Mode     : cuda", local_rank)

    effective_batch = cfg.per_gpu_batch * world_size * cfg.grad_accum_steps
    log(f"  Per-GPU Batch       : {cfg.per_gpu_batch}", local_rank)
    log(f"  Gradient Accum Steps: {cfg.grad_accum_steps}", local_rank)
    log(f"  Effective Batch Size: {effective_batch} (across {world_size} GPUs)", local_rank)

    # 3. Optimizer & Scheduler
    optimizer = Lion(model.parameters(), lr=cfg.learning_rate)
    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps)

    # 4. Critics (Dedicated copy per GPU)
    log(f"  Loading WavLM       : {cfg.wavlm_dir}", local_rank)
    wavlm = WavLMModel.from_pretrained(cfg.wavlm_dir).to(device)
    slm_loss_fn = SLMLoss(wavlm)

    log(f"  Loading IndicWhisper: {cfg.whisper_dir}", local_rank)
    whisper_enc = WhisperModel.from_pretrained(cfg.whisper_dir).encoder.to(device)
    whisper_ext = WhisperFeatureExtractor.from_pretrained(cfg.whisper_dir)
    srfd_loss_fn = SRFDLoss(whisper_enc, whisper_ext)

    # 5. Dataset & Distributed Samplers
    log(f"  Dataset             : {cfg.dataset_dir}", local_rank)
    log("Loading Shrutilipi dataset...", local_rank)
    full_ds = load_dataset("parquet", data_dir=cfg.dataset_dir, split="train")
    split = full_ds.train_test_split(test_size=cfg.val_split, seed=42)

    log(f"  Train samples       : {len(split['train'])}", local_rank)
    log(f"  Val samples         : {len(split['test'])}", local_rank)

    train_ds = ShrutilipiDataset(
        split["train"], max_audio_len=cfg.max_audio_len, max_text_len=cfg.max_text_len,
        mel_channels=cfg.mel_channels, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )
    val_ds = ShrutilipiDataset(
        split["test"], max_audio_len=cfg.max_audio_len, max_text_len=cfg.max_text_len,
        mel_channels=cfg.mel_channels, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True) if is_distributed else None
    val_sampler   = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False) if is_distributed else None

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

    batches_per_gpu = len(train_loader)
    opt_steps_per_epoch = batches_per_gpu // cfg.grad_accum_steps
    total_epochs = (cfg.total_steps // max(1, opt_steps_per_epoch)) + 1

    log(f"  Batches/GPU/epoch   : {batches_per_gpu}", local_rank)
    log(f"  Opt steps/epoch     : {opt_steps_per_epoch}", local_rank)
    log(f"  Val batches         : {len(val_loader)}", local_rank)
    log(f"  Total target steps  : {cfg.total_steps}", local_rank)
    log(f"  Epochs needed       : {total_epochs}", local_rank)
    log("=" * 60, local_rank)

    # 6. Resume
    global_step = 0
    best_val_loss = float("inf")
    resume_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
    if os.path.exists(resume_path):
        global_step = load_checkpoint(resume_path, model, optimizer, scheduler)

    # 7. Training Loop
    model.train()
    for epoch in range(total_epochs):
        if global_step >= cfg.total_steps:
            break

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(
            enumerate(train_loader), total=batches_per_gpu,
            desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch", ncols=140,
            disable=not is_main_process(local_rank),
        )

        optimizer.zero_grad()

        for batch_idx, (text_tokens, ref_mel, real_audio) in pbar:
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

            scaled_loss = total_loss / cfg.grad_accum_steps
            scaled_loss.backward()

            # Optimizer Step on Accumulation
            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main_process(local_rank):
                    pbar.set_postfix({
                        "step": global_step,
                        "loss": f"{total_loss.item():.3f}",
                        "mel": f"{loss_mel.item():.3f}",
                        "slm": f"{loss_slm.item():.3f}",
                        "dur": f"{loss_dur.item():.3f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                    })

                # Checkpoint (Rank 0 only)
                if is_main_process(local_rank) and global_step % cfg.save_every == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, total_loss.item(),
                                    os.path.join(cfg.checkpoint_dir, "latest.pt"))
                    save_checkpoint(model, optimizer, scheduler, global_step, total_loss.item(),
                                    os.path.join(cfg.checkpoint_dir, f"step_{global_step}.pt"))
                    print(f"\n  [Checkpoint] Saved at step {global_step}")

                # Validation (Rank 0 only)
                if is_main_process(local_rank) and global_step % cfg.val_every == 0:
                    val_mel, val_srfd = validate(model, val_loader, srfd_loss_fn, device, global_step)
                    print(f"\n  [Val @ step {global_step}] Mel L1: {val_mel:.4f} | SR-FD: {val_srfd:.4f}")
                    if val_mel < best_val_loss:
                        best_val_loss = val_mel
                        save_checkpoint(model, optimizer, scheduler, global_step, val_mel,
                                        os.path.join(cfg.checkpoint_dir, "best.pt"))
                        print(f"  [Val] New best model saved! (Mel L1: {val_mel:.4f})")

                if is_distributed:
                    dist.barrier()

    if is_main_process(local_rank):
        save_checkpoint(model, optimizer, scheduler, global_step, total_loss.item(),
                        os.path.join(cfg.checkpoint_dir, "final.pt"))
        print("\n" + "=" * 60)
        print("  Training Complete!")
        print(f"  Final step: {global_step}")
        print(f"  Best val Mel L1: {best_val_loss:.4f}")
        print("=" * 60)

    if is_distributed:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main Launcher
# ---------------------------------------------------------------------------

def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="TamilTTS DDP Training")
    parser.add_argument("--dataset_dir", type=str, default=cfg.dataset_dir)
    parser.add_argument("--wavlm_dir", type=str, default=cfg.wavlm_dir)
    parser.add_argument("--whisper_dir", type=str, default=cfg.whisper_dir)
    parser.add_argument("--checkpoint_dir", type=str, default=cfg.checkpoint_dir)
    parser.add_argument("--per_gpu_batch", type=int, default=cfg.per_gpu_batch)
    parser.add_argument("--grad_accum_steps", type=int, default=cfg.grad_accum_steps)
    parser.add_argument("--total_steps", type=int, default=cfg.total_steps)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg.dataset_dir = args.dataset_dir
    cfg.wavlm_dir = args.wavlm_dir
    cfg.whisper_dir = args.whisper_dir
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.per_gpu_batch = args.per_gpu_batch
    cfg.grad_accum_steps = args.grad_accum_steps
    cfg.total_steps = args.total_steps

    if args.resume:
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        import shutil
        shutil.copy(args.resume, os.path.join(cfg.checkpoint_dir, "latest.pt"))

    # Case 1: Launched via torchrun
    if "LOCAL_RANK" in os.environ or "RANK" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        train_worker(local_rank, world_size, cfg)

    # Case 2: Launched via regular `python train.py` on Multi-GPU machine -> Auto-spawn DDP
    elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()
        print(f"🔥 Detected {world_size} GPUs! Auto-spawning DDP multi-process training...")
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        mp.spawn(train_worker, nprocs=world_size, args=(world_size, cfg))

    # Case 3: Single GPU
    else:
        train_worker(0, 1, cfg)


if __name__ == "__main__":
    main()
