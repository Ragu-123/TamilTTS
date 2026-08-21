"""
TamilTTS Multi-GPU DDP Training Pipeline — v7 (SOTA Kokoro / AI4Bharat Architecture)
====================================================================================
Features:
- Decoupled Acoustic Model (Transformer + MAS + Style Diffusion + 5-layer PostNet).
- Pre-trained Frozen Universal HiFi-GAN Vocoder (zero buzzing, zero compute waste).
- Monotonic Alignment Search (MAS) for dynamic phoneme-to-mel frame duration learning.
- Dual Mel-Spectrogram Loss (Pre-PostNet coarse + Post-PostNet refined).
- Masked Log-Duration Loss (Kokoro-82M / FastPitch standard).
- Multi-GPU DDP support with gradient accumulation, Lion optimizer, and grad explosion guard.
- Low-memory row-group Parquet streaming (0 MB disk space overhead).
"""
import os
import gc
import argparse
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lion_pytorch import Lion
from transformers import WavLMModel, WhisperModel, WhisperFeatureExtractor

from config import Config
from models import TamilTTS
from models.vocoder import load_pretrained_vocoder
from models.alignment import monotonic_alignment_search
from losses import DualMelLoss, LogDurationLoss, SLMLoss, SRFDLoss
from data import build_tamil_datasets
from utils import save_checkpoint, load_checkpoint, count_parameters, get_lr_scheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_main_process(rank):
    return rank == 0 or rank == -1


def log(msg, rank=0):
    if is_main_process(rank):
        print(msg)


# ---------------------------------------------------------------------------
# Evaluation Function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, srfd_loss_fn, device, local_rank=0):
    was_training = model.training
    model.eval()

    raw_model = model.module if hasattr(model, "module") else model
    total_mel_loss = 0.0
    total_srfd = 0.0
    n = 0

    pbar = tqdm(val_loader, desc="Validating", leave=False, disable=not is_main_process(local_rank))
    for text_tokens, ref_mel, real_audio in pbar:
        text_tokens = text_tokens.to(device)
        ref_mel     = ref_mel.to(device)
        real_audio  = real_audio.to(device)
        text_mask   = (text_tokens == 0)

        gen_audio, mel_refined, mel_coarse, dur_pred, _ = raw_model(
            text_tokens, ref_mel,
            text_mask=text_mask,
        )

        mel_target = ref_mel.transpose(1, 2)
        min_t = min(mel_refined.size(1), mel_target.size(1))
        mel_loss = F.l1_loss(mel_refined[:, :min_t], mel_target[:, :min_t])
        total_mel_loss += mel_loss.item()

        min_a = min(gen_audio.size(1), real_audio.size(1))
        srfd = srfd_loss_fn(real_audio[:, :min_a], gen_audio[:, :min_a])
        total_srfd += srfd.item()

        n += 1
        pbar.set_postfix(mel=f"{mel_loss.item():.4f}", srfd=f"{srfd.item():.4f}")

    if was_training:
        model.train()
    return total_mel_loss / max(n, 1), total_srfd / max(n, 1)


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
    log("  TamilTTS Training Pipeline (v7 — SOTA Kokoro / AI4Bharat)", local_rank)
    log("=" * 60, local_rank)

    # 2. Model & Pre-trained Frozen Vocoder
    model = TamilTTS(cfg).to(device)

    # Load frozen universal HiFi-GAN weights into the vocoder submodule
    if cfg.vocoder_ckpt and os.path.exists(cfg.vocoder_ckpt):
        log(f"  Loading Frozen Vocoder: {cfg.vocoder_ckpt}", local_rank)
        vocoder_loaded = load_pretrained_vocoder(device=device, checkpoint_path=cfg.vocoder_ckpt)
        model.vocoder.load_state_dict(vocoder_loaded.state_dict(), strict=False)
        # Freeze vocoder parameters
        for p in model.vocoder.parameters():
            p.requires_grad = False

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

    # 3. Optimizer & Scheduler (Only trains trainable acoustic parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Lion(trainable_params, lr=cfg.learning_rate)
    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps)

    # 4. Loss Functions
    dual_mel_loss_fn = DualMelLoss(coarse_weight=cfg.weight_mel_coarse, refined_weight=cfg.weight_mel_refined)
    log_dur_loss_fn = LogDurationLoss()

    log(f"  Loading WavLM       : {cfg.wavlm_dir}", local_rank)
    wavlm = WavLMModel.from_pretrained(cfg.wavlm_dir, low_cpu_mem_usage=True).to(device)
    slm_loss_fn = SLMLoss(wavlm)

    log(f"  Loading IndicWhisper: {cfg.whisper_dir}", local_rank)
    whisper_model = WhisperModel.from_pretrained(cfg.whisper_dir, low_cpu_mem_usage=True)
    whisper_enc = whisper_model.encoder.to(device)
    whisper_ext = WhisperFeatureExtractor.from_pretrained(cfg.whisper_dir)
    srfd_loss_fn = SRFDLoss(whisper_enc, whisper_ext)
    del whisper_model
    gc.collect()

    # 5. Dataset & Distributed Samplers
    log(f"  Dataset             : {cfg.dataset_dir}", local_rank)
    train_ds, val_ds = build_tamil_datasets(cfg.dataset_dir, cfg)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True) if is_distributed else None
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=local_rank, shuffle=False) if is_distributed else None

    use_workers = cfg.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
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

    gc.collect()
    torch.cuda.empty_cache()

    # 7. Training Loop
    try:
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

                # Mask padding text tokens (0 is PAD)
                text_mask = (text_tokens == 0)

                # Forward pass: Dynamically aligns text to mel via Monotonic Alignment Search (MAS)
                gen_audio, mel_refined, mel_coarse, dur_pred, log_dur_pred = model(
                    text_tokens, ref_mel,
                    target_mel_len=target_mel_len,
                    text_mask=text_mask,
                )

                # Dynamic Duration Targets from MAS
                with torch.no_grad():
                    raw_model = model.module if hasattr(model, "module") else model
                    text_emb = raw_model.text_encoder(text_tokens, mask=text_mask)
                    mel_proj = raw_model.mel_align_proj(ref_mel.transpose(1, 2))
                    target_dur, _ = monotonic_alignment_search(text_emb, mel_proj, text_mask=text_mask)

                # --- Losses ---
                # 1. Dual Mel Spectrogram Loss (Coarse + 5-layer PostNet Refined)
                mel_target = ref_mel.transpose(1, 2)
                loss_mel, loss_ref, loss_crs = dual_mel_loss_fn(mel_refined, mel_coarse, mel_target)

                # 2. Log-Duration Loss (Huber / MSE in log-space)
                loss_dur = log_dur_loss_fn(log_dur_pred, target_dur, mask=text_mask)

                # 3. WavLM Speech Language Model Perceptual Loss
                min_a = min(gen_audio.size(1), real_audio.size(1))
                loss_slm = slm_loss_fn(real_audio[:, :min_a], gen_audio[:, :min_a])

                total_loss = (
                    45.0 * loss_mel
                    + cfg.weight_dur * loss_dur
                    + cfg.weight_slm * loss_slm
                )

                scaled_loss = total_loss / cfg.grad_accum_steps
                scaled_loss.backward()

                # Optimizer Step on Accumulation
                if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    grad_val = grad_norm.item() if torch.isfinite(grad_norm) else 0.0

                    # Kokoro-82M Gradient Explosion Guard: Skip corrupt batches
                    if not torch.isfinite(grad_norm) or grad_val > 10.0:
                        log(f"⚠️ [Step {global_step}] Gradient explosion (norm={grad_val:.2f} > 10.0). Skipping batch to prevent weight corruption.", local_rank)
                        optimizer.zero_grad()
                        continue

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if is_main_process(local_rank):
                        pbar.set_postfix({
                            "step": global_step,
                            "loss": f"{total_loss.item():.3f}",
                            "mel": f"{loss_ref.item():.3f}",
                            "dur": f"{loss_dur.item():.3f}",
                            "slm": f"{loss_slm.item():.3f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                        })

                    # Periodic Validation & Checkpointing (Rank 0 only)
                    if global_step % cfg.save_every == 0 and is_main_process(local_rank):
                        val_mel, val_srfd = evaluate(model, val_loader, srfd_loss_fn, device, local_rank)
                        log(f"\n[Step {global_step}] Val Mel Loss: {val_mel:.4f} | Val SR-FD: {val_srfd:.4f}")

                        save_checkpoint(
                            os.path.join(cfg.checkpoint_dir, f"step_{global_step}.pt"),
                            model, optimizer, scheduler, global_step, val_mel
                        )
                        save_checkpoint(
                            os.path.join(cfg.checkpoint_dir, "latest.pt"),
                            model, optimizer, scheduler, global_step, val_mel
                        )

                        if val_mel < best_val_loss:
                            best_val_loss = val_mel
                            save_checkpoint(
                                os.path.join(cfg.checkpoint_dir, "best.pt"),
                                model, optimizer, scheduler, global_step, val_mel
                            )
                            log(f"  🏆 New Best Model Saved (Mel Loss: {best_val_loss:.4f})")

            # End-of-Epoch Evaluation (Rank 0 only)
            if is_main_process(local_rank):
                val_mel, val_srfd = evaluate(model, val_loader, srfd_loss_fn, device, local_rank)
                log(f"\nEpoch {epoch+1} Complete | Val Mel Loss: {val_mel:.4f} | Val SR-FD: {val_srfd:.4f}")

                save_checkpoint(
                    os.path.join(cfg.checkpoint_dir, "latest.pt"),
                    model, optimizer, scheduler, global_step, val_mel
                )
                if val_mel < best_val_loss:
                    best_val_loss = val_mel
                    save_checkpoint(
                        os.path.join(cfg.checkpoint_dir, "best.pt"),
                        model, optimizer, scheduler, global_step, val_mel
                    )

    finally:
        # Proper DDP teardown
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TamilTTS Training Pipeline (v7 — Kokoro SOTA)")
    parser.add_argument("--dataset_dir", type=str, nargs="+", default=None,
                        help="Path(s) to dataset folder(s) containing Parquet files")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--per_gpu_batch", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--vocoder_ckpt", type=str, default=None,
                        help="Path to pre-trained frozen HiFi-GAN generator weights")
    args = parser.parse_args()

    cfg = Config()
    if args.dataset_dir:
        cfg.dataset_dir = args.dataset_dir if len(args.dataset_dir) > 1 else args.dataset_dir[0]
    if args.total_steps:
        cfg.total_steps = args.total_steps
    if args.per_gpu_batch:
        cfg.per_gpu_batch = args.per_gpu_batch
    if args.grad_accum_steps:
        cfg.grad_accum_steps = args.grad_accum_steps
    if args.lr:
        cfg.learning_rate = args.lr
    if args.vocoder_ckpt:
        cfg.vocoder_ckpt = args.vocoder_ckpt

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Detect DDP environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        # Multi-GPU DDP process
        train_worker(local_rank, world_size, cfg)
    else:
        # Check available GPUs for auto-spawning DDP
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"🔥 Auto-spawning DDP multi-process training across {num_gpus} GPUs...")
            torch.multiprocessing.spawn(
                train_worker,
                args=(num_gpus, cfg),
                nprocs=num_gpus,
                join=True
            )
        else:
            train_worker(0, 1, cfg)


if __name__ == "__main__":
    main()
