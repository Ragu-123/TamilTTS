"""
TamilTTS Multi-GPU Training Pipeline — SOTA FastPitch + RAD-TTS Architecture
============================================================================
Features:
- Dynamic Batch Collation (no rigid sequence padding).
- Learned RAD-TTS Alignment Network (Forward-Sum Loss + Binarization Loss + Viterbi Durations).
- Masked Dual Mel Loss (Coarse + 5-Layer PostNet Refined Mel at 22.05 kHz).
- Frozen Universal HiFi-GAN Vocoder (13.93M parameters).
- Staged Training (Stage 1: Mel + Alignment + Duration; Stage 2: Prosody + SLM).
- Hardware Auto-Optimization (RTX PRO 6000 Blackwell 96GB or 4x L4 DDP).
"""
import os
import gc
import argparse
from contextlib import nullcontext
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import WavLMModel, WhisperModel, WhisperFeatureExtractor

from config import Config
from models import TamilTTS
from models.vocoder import load_pretrained_vocoder
from losses import DualMelLoss, LogDurationLoss, SLMLoss, SRFDLoss, PitchLoss
from data import build_tamil_datasets
from data.dataset import tamil_tts_collate_fn
from utils import save_checkpoint, load_checkpoint, count_parameters, get_lr_scheduler


def is_main_process(rank):
    return rank == 0 or rank == -1


def log(msg, rank=0):
    if is_main_process(rank):
        print(msg)


def auto_configure_hardware(cfg):
    if not torch.cuda.is_available():
        return "cpu", 1

    num_gpus = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("=" * 60)
    print("  HARDWARE AUTO-DETECTION & OPTIMIZATION")
    print("=" * 60)
    print(f"  Detected GPUs  : {num_gpus}x {gpu_name}")
    print(f"  Primary VRAM   : {total_vram_gb:.1f} GB")

    if num_gpus == 1 and total_vram_gb >= 40.0:
        batch_sz = 64 if total_vram_gb >= 70.0 else 32
        print(f"  🚀 ULTRA-FAST HIGH-VRAM MODE ({gpu_name})")
        print(f"  -> Batch Size: {batch_sz} | Grad Accum: 1 (Instant single-step updates, 0 DDP overhead)")
        cfg.per_gpu_batch = batch_sz
        cfg.grad_accum_steps = 1
        return "single_high_vram", 1

    elif num_gpus > 1:
        print(f"  🔥 MULTI-GPU DDP MODE ({num_gpus}x {gpu_name})")
        print(f"  -> Per-GPU Batch: {cfg.per_gpu_batch} | Grad Accum: {cfg.grad_accum_steps}")
        return "ddp", num_gpus

    else:
        print(f"  ⚡ STANDARD SINGLE GPU MODE ({gpu_name})")
        cfg.per_gpu_batch = 16 if total_vram_gb >= 15.0 else 8
        cfg.grad_accum_steps = 4
        return "single_gpu", 1


@torch.no_grad()
def evaluate(model, val_loader, srfd_loss_fn, device, local_rank=0):
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()
    dual_mel_fn = DualMelLoss(coarse_weight=0.5, refined_weight=1.0)

    total_mel_loss = 0.0
    total_srfd = 0.0
    count = 0

    pbar = tqdm(val_loader, desc="Validating", unit="batch", ncols=120, disable=not is_main_process(local_rank))
    for batch in pbar:
        if batch is None:
            continue
        text_tokens, text_lens, ref_mel, mel_lens, real_audio, audio_lens = batch[:6]
        target_dur = batch[6].to(device).float() if len(batch) >= 7 and batch[6] is not None else None

        text_tokens = text_tokens.to(device)
        text_lens   = text_lens.to(device)
        ref_mel     = ref_mel.to(device)
        mel_lens    = mel_lens.to(device)
        real_audio  = real_audio.to(device)

        with torch.no_grad():
            gen_audio, mel_refined, mel_coarse, _, _, _, _, _, _ = eval_model(
                text_tokens, text_lens,
                ref_mel=ref_mel,
                mel_lens=mel_lens,
                target_dur=target_dur,
                return_audio=(srfd_loss_fn is not None)
            )
            mel_target = ref_mel.transpose(1, 2)
            loss_mel, _, _ = dual_mel_fn(mel_refined, mel_coarse, mel_target, mel_lens=mel_lens)

            srfd = torch.tensor(0.0, device=device)
            if srfd_loss_fn is not None and gen_audio is not None:
                min_a = min(gen_audio.size(1), real_audio.size(1))
                srfd = srfd_loss_fn(real_audio[:, :min_a], gen_audio[:, :min_a])

        total_mel_loss += loss_mel.item()
        total_srfd += srfd.item()
        count += 1

        if is_main_process(local_rank):
            pbar.set_postfix({"mel": f"{loss_mel.item():.4f}", "srfd": f"{srfd.item():.4f}"})

    eval_model.train()
    return total_mel_loss / max(count, 1), total_srfd / max(count, 1)


def train_worker(local_rank, world_size, cfg):
    is_distributed = world_size > 1
    if is_distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
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
    log("  TamilTTS Training Pipeline (SOTA FastPitch + RAD-TTS Standard)", local_rank)
    log("=" * 60, local_rank)

    # 1. Model & Vocoder
    model = TamilTTS(cfg).to(device)

    if cfg.vocoder_ckpt and os.path.exists(cfg.vocoder_ckpt):
        log(f"  Loading Frozen Vocoder: {cfg.vocoder_ckpt}", local_rank)
        vocoder_loaded = load_pretrained_vocoder(device=device, checkpoint_path=cfg.vocoder_ckpt)
        model.vocoder.load_state_dict(vocoder_loaded.state_dict(), strict=False)
        for p in model.vocoder.parameters():
            p.requires_grad = False

    total_p, train_p = count_parameters(model)
    log(f"  Total Parameters    : {total_p / 1e6:.2f}M", local_rank)
    log(f"  Trainable Parameters: {train_p / 1e6:.2f}M", local_rank)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        log(f"  🔥 DDP Active       : {world_size} GPUs (Rank {local_rank} on cuda:{local_rank})", local_rank)
    else:
        log(f"  🚀 GPU Active       : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", local_rank)

    effective_batch = cfg.per_gpu_batch * world_size * cfg.grad_accum_steps
    log(f"  Per-GPU Batch       : {cfg.per_gpu_batch}", local_rank)
    log(f"  Gradient Accum Steps: {cfg.grad_accum_steps}", local_rank)
    log(f"  Effective Batch Size: {effective_batch}", local_rank)

    # 2. Optimizer & Scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, cfg.total_steps)

    # 3. Loss Functions
    dual_mel_loss_fn = DualMelLoss(coarse_weight=cfg.weight_mel_coarse, refined_weight=cfg.weight_mel_refined)
    log_dur_loss_fn  = LogDurationLoss()
    pitch_loss_fn    = PitchLoss()
    srfd_loss_fn = None
    if getattr(cfg, "whisper_dir", None) and os.path.exists(cfg.whisper_dir):
        log(f"  Loading IndicWhisper: {cfg.whisper_dir}", local_rank)
        whisper_model = WhisperModel.from_pretrained(cfg.whisper_dir, low_cpu_mem_usage=True)
        whisper_enc = whisper_model.encoder.to(device)
        whisper_ext = WhisperFeatureExtractor.from_pretrained(cfg.whisper_dir)
        srfd_loss_fn = SRFDLoss(whisper_enc, whisper_ext, sample_rate=cfg.sample_rate)
        del whisper_model
        gc.collect()

    slm_loss_fn = None
    if cfg.weight_slm > 0.0:
        log(f"  Loading WavLM       : {cfg.wavlm_dir}", local_rank)
        wavlm = WavLMModel.from_pretrained(cfg.wavlm_dir, low_cpu_mem_usage=True).to(device)
        slm_loss_fn = SLMLoss(wavlm, sample_rate=cfg.sample_rate)

    # 4. Dataset & Dynamic Loaders
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
        collate_fn=tamil_tts_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_workers,
        prefetch_factor=3 if use_workers else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=tamil_tts_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_workers,
        prefetch_factor=3 if use_workers else None,
        drop_last=False,
    )

    batches_per_gpu = len(train_loader)
    opt_steps_per_epoch = max(1, batches_per_gpu // cfg.grad_accum_steps)
    total_epochs = (cfg.total_steps // opt_steps_per_epoch) + 1

    log(f"  Batches/epoch       : {batches_per_gpu}", local_rank)
    log(f"  Opt steps/epoch     : {opt_steps_per_epoch}", local_rank)
    log(f"  Val batches         : {len(val_loader)}", local_rank)
    log(f"  Total target steps  : {cfg.total_steps}", local_rank)
    log(f"  Epochs needed       : {total_epochs}", local_rank)
    log("=" * 60, local_rank)

    # 5. Checkpoint Resume
    global_step = 0
    best_val_loss = float("inf")
    resume_path = getattr(cfg, "resume_path", None) or os.path.join(cfg.checkpoint_dir, "latest.pt")
    if resume_path and os.path.exists(resume_path):
        global_step = load_checkpoint(resume_path, model, optimizer, scheduler)

    gc.collect()
    torch.cuda.empty_cache()

    # 6. Training Loop
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

            for batch_idx, batch in pbar:
                if batch is None or global_step >= cfg.total_steps:
                    continue

                if len(batch) >= 7:
                    text_tokens, text_lens, ref_mel, mel_lens, real_audio, audio_lens, batch_gt_dur = batch
                    batch_gt_dur = batch_gt_dur.to(device)
                else:
                    text_tokens, text_lens, ref_mel, mel_lens, real_audio, audio_lens = batch
                    batch_gt_dur = None

                text_tokens = text_tokens.to(device)
                text_lens   = text_lens.to(device)
                ref_mel     = ref_mel.to(device)
                mel_lens    = mel_lens.to(device)
                real_audio  = real_audio.to(device)

                # Ground-truth duration targets (IndicMFA Ground-Truth with Proportional Fallback)
                target_dur = None
                if getattr(cfg, "use_gt_durations", True):
                    if batch_gt_dur is not None and (batch_gt_dur > 0).any():
                        target_dur = batch_gt_dur.float()
                    else:
                        target_dur = torch.zeros_like(text_tokens, dtype=torch.float32)
                        for b_i in range(text_tokens.size(0)):
                            t_len = max(int(text_lens[b_i].item()), 1)
                            m_len = max(int(mel_lens[b_i].item()), 1)
                            base_d = m_len / t_len
                            target_dur[b_i, :t_len] = base_d

                is_accumulating = is_distributed and ((batch_idx + 1) % cfg.grad_accum_steps != 0)
                sync_context = model.no_sync() if is_accumulating else nullcontext()
                with sync_context:
                    need_audio = (slm_loss_fn is not None and cfg.weight_slm > 0.0)
                    gen_audio, mel_refined, mel_coarse, dur_pred, log_dur_pred, align_dur, fwd_loss, bin_loss, pred_f0 = model(
                        text_tokens, text_lens,
                        ref_mel=ref_mel,
                        mel_lens=mel_lens,
                        target_dur=target_dur,
                        return_audio=need_audio,
                    )

                    # --- Losses ---
                    # 1. Masked Dual Mel Loss (computed strictly on valid speech frames)
                    mel_target = ref_mel.transpose(1, 2)
                    loss_mel, loss_ref, loss_crs = dual_mel_loss_fn(
                        mel_refined, mel_coarse, mel_target, mel_lens=mel_lens
                    )

                    # 2. Masked Log-Duration Loss (supervised by alignment-derived durations)
                    loss_dur = log_dur_loss_fn(log_dur_pred, align_dur, text_lens=text_lens)

                    # 3. RAD-TTS Forward-Sum + Warmed-up Binarization Loss
                    bin_warmup = getattr(cfg, "bin_warmup_steps", 5000)
                    bin_scale = min(1.0, max(0.0, global_step / max(bin_warmup, 1)))
                    loss_align = fwd_loss + (cfg.weight_bin * bin_scale) * bin_loss

                    # 4. Optional Stage 2 SLM Loss
                    loss_slm = torch.tensor(0.0, device=device)
                    if slm_loss_fn is not None and cfg.weight_slm > 0.0:
                        min_a = min(gen_audio.size(1), real_audio.size(1))
                        loss_slm = slm_loss_fn(real_audio[:, :min_a], gen_audio[:, :min_a])

                    total_loss = (
                        45.0 * loss_mel
                        + cfg.weight_dur * loss_dur
                        + cfg.weight_align * loss_align
                        + cfg.weight_slm * loss_slm
                    )

                    scaled_loss = total_loss / cfg.grad_accum_steps
                    scaled_loss.backward()

                # Optimizer Step
                if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

                    if not torch.isfinite(grad_norm):
                        log(f"⚠️ [Step {global_step}] Non-finite gradient detected (NaN/Inf). Skipping batch.", local_rank)
                        optimizer.zero_grad()
                        continue

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if is_main_process(local_rank):
                        pbar.set_postfix({
                            "step": global_step,
                            "loss": f"{total_loss.item():.2f}",
                            "mel": f"{loss_ref.item():.3f}",
                            "align": f"{fwd_loss.item():.2f}",
                            "dur": f"{loss_dur.item():.3f}",
                            "lr": f"{scheduler.get_last_lr()[0]:.1e}",
                        })

                    # Checkpoint validation
                    if global_step % cfg.save_every == 0:
                        val_mel, val_srfd = evaluate(model, val_loader, srfd_loss_fn, device, local_rank)
                        if is_main_process(local_rank):
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
                        if is_distributed:
                            dist.barrier()

            # End-of-Epoch Evaluation
            val_mel, val_srfd = evaluate(model, val_loader, srfd_loss_fn, device, local_rank)
            if is_main_process(local_rank):
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
            if is_distributed:
                dist.barrier()

    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="TamilTTS Training Pipeline (SOTA FastPitch + RAD-TTS)")
    parser.add_argument("--dataset_dir", nargs="+", default=None, help="Dataset directories")
    parser.add_argument("--vocoder_ckpt", type=str, default=None, help="Path to pre-trained frozen HiFi-GAN generator.pt")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--steps", type=int, default=None, help="Total training steps override")
    args = parser.parse_args()

    cfg = Config()
    if args.resume:
        cfg.resume_path = args.resume
    if args.dataset_dir:
        cfg.dataset_dir = args.dataset_dir
    if args.vocoder_ckpt:
        cfg.vocoder_ckpt = args.vocoder_ckpt
    if args.batch_size:
        cfg.per_gpu_batch = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.steps:
        cfg.total_steps = args.steps

    train_mode, num_devices = auto_configure_hardware(cfg)

    if train_mode == "ddp":
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        print(f"🔥 Spawning DDP multi-process training across {num_devices} GPUs...")
        torch.multiprocessing.spawn(
            train_worker,
            args=(num_devices, cfg),
            nprocs=num_devices,
            join=True
        )
    else:
        train_worker(0, 1, cfg)


if __name__ == "__main__":
    main()
