"""
TamilTTSv2 Staged Multi-GPU Training Pipeline
=============================================
FastPitch-variant acoustic model + FiLM style conditioning + frozen 22.05 kHz vocoder.

Staged objectives (by global_step):
  Stage 1 (0 .. stage1_steps)            : masked dual-mel + duration + pitch/energy regression.
  Stage 2 (>= stage1_steps)              : + LSGAN adversarial + feature matching through MPD & MRD
                                           (discriminator step BEFORE generator step, detached audio).
  Stage 3 (>= slm_start_step)            : + WavLM SLM perceptual loss, linear ramp over slm_ramp_steps
                                           up to weight_slm_final.

Key behaviors:
  - GT prosody (log-F0/voicing, energy) is passed during training; the model mixes
    GT and predicted conditioning 50/50 (one shared coin flip per step) so the
    decoder also sees the predicted-prosody distribution it gets at inference.
  - GAN/SLM losses consume per-sample random crops aligned to true audio lengths
    (no collate zero-padding enters the discriminators).
  - Reference conditioning uses torch.roll(mel, 1, dim=0): every utterance is styled by a
    DIFFERENT utterance to prevent style leakage.
  - Two optimizers: AdamW generator (cosine warmup schedule) + AdamW discriminator (betas 0.8/0.99).
  - bf16 autocast on CUDA (no scaler needed), TF32 enabled.
  - EMA of acoustic weights; validation samples synthesized with EMA weights.
  - Spawn-based DDP for Kaggle 4xL4 (`python train.py`, no torchrun required).
"""
import os
import gc
import argparse
import warnings
from contextlib import nullcontext

# Silence third-party deprecation noise (pyworld -> pkg_resources) before any
# heavy import; module-level so DDP-spawned ranks inherit it on fresh import.
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from config import Config
from models.tamil_tts_v2 import TamilTTSv2
from models.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from losses import (
    MelLoss,
    DurationLoss,
    PitchEnergyLoss,
    DiscriminatorLoss,
    GeneratorAdversarialLoss,
    FeatureMatchingLoss,
    SLMLoss,
)
from data import build_tamil_datasets
from data.dataset import tamil_tts_collate_fn, RowGroupBatchSampler
from utils import (
    EMA,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    get_lr_scheduler,
    unwrap_model,
)


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
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

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
        print(f"  -> Batch Size: {batch_sz} | Grad Accum: 1")
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
        cfg.grad_accum_steps = max(1, cfg.grad_accum_steps)
        return "single_gpu", 1


def amp_context(device, cfg):
    if device.type == "cuda" and getattr(cfg, "use_bf16", False):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def to_device_batch(batch, device):
    return {
        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def run_discriminators(mpd, mrd, audio):
    scores_mpd, feats_mpd = mpd(audio)
    scores_mrd, feats_mrd = mrd(audio)
    scores = list(scores_mpd) + list(scores_mrd)
    feats = [feats_mpd, feats_mrd]
    return scores, feats


def make_gan_segments(vocoder, mel_pred, mel_lens, real_audio, hop_length, seg_frames=32):
    """HiFi-GAN-style matched segment sampling (bounds GAN-stage memory).

    Picks one random `seg_frames` window per sample from the PREDICTED mel
    (bounded by each sample's true mel_len, so zero padding enters), vocodes
    ONLY that window with grad flowing back to mel_pred, and cuts the REAL
    waveform at the aligned sample offset. Peak memory is capped at
    seg_frames*hop = 8192 samples regardless of utterance length — vocoding
    full padded sequences (B x up-to-10s audio through 256x upsampling)
    exceeded 22GB at the first GAN step and killed a rank, deadlocking the
    rest on the next collective.

    Returns (real_seg [B, seg_frames*hop], fake_seg [B, seg_frames*hop]).
    """
    B = mel_pred.size(0)
    device = mel_pred.device
    # A clip's mel frame count can exceed audio_len/hop (center-pad framing adds
    # frames), so window starts are bounded by BOTH axes; otherwise the audio
    # gather can read past the padded waveform -> CUDA device assert.
    lens = mel_lens.to(device).clamp(min=seg_frames)
    max_start = max(0, (real_audio.size(-1) - seg_frames * hop_length) // hop_length)
    lens = torch.minimum(lens, torch.full_like(lens, max_start + seg_frames)).clamp(min=seg_frames)
    starts = (torch.rand(B, device=device) * (lens - seg_frames + 1)).floor().long()
    starts = torch.minimum(starts, torch.full_like(starts, max_start))

    mel_t = mel_pred.transpose(1, 2)                     # [B, 80, Tm]
    idx_m = starts.view(B, 1, 1) + torch.arange(seg_frames, device=device).view(1, 1, -1)
    mel_win = mel_t.gather(2, idx_m.expand(-1, mel_t.size(1), -1))      # [B, 80, L]

    La = seg_frames * hop_length
    idx_a = (starts * hop_length).view(B, 1) + torch.arange(La, device=device).view(1, -1)
    # clamp guards clips shorter than one window (start=0 still overruns); the
    # repeated tail values act as zero-amplitude-ish padding for those rare rows.
    idx_a = idx_a.clamp(max=real_audio.size(1) - 1)
    real_seg = real_audio.gather(1, idx_a)               # [B, La]

    fake_seg = vocoder(mel_win)                          # [B, La], grad -> mel_pred
    return real_seg.float(), fake_seg.float()


def slm_weight(step, cfg):
    if step < cfg.slm_start_step:
        return 0.0
    progress = (step - cfg.slm_start_step + 1) / float(max(cfg.slm_ramp_steps, 1))
    return min(cfg.weight_slm_final, cfg.weight_slm_final * progress)


def build_slm(device, cfg, cache):
    if cache.get("unavailable"):
        return None
    if cache.get("loss") is None:
        try:
            from transformers import WavLMModel
            log(f"  Loading WavLM for SLM loss: {cfg.wavlm_dir}")
            wavlm = WavLMModel.from_pretrained(cfg.wavlm_dir, low_cpu_mem_usage=True).to(device).eval()
            cache["loss"] = SLMLoss(wavlm, sample_rate=cfg.sample_rate)
            gc.collect()
        except Exception as exc:
            log(f"  ⚠️ SLM loss disabled ({exc})")
            cache["unavailable"] = True
    return cache.get("loss")


def build_srfd_bundle(device, cfg):
    """Optional Whisper-based SR-FD validation metric; returns None when unavailable."""
    try:
        if not (getattr(cfg, "whisper_dir", None) and os.path.exists(cfg.whisper_dir)):
            return None
        from transformers import WhisperModel, WhisperFeatureExtractor
        import torchaudio.transforms as T
        log(f"  Loading IndicWhisper for SR-FD validation: {cfg.whisper_dir}")
        whisper_model = WhisperModel.from_pretrained(cfg.whisper_dir, low_cpu_mem_usage=True)
        encoder = whisper_model.encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad = False
        extractor = WhisperFeatureExtractor.from_pretrained(cfg.whisper_dir)
        resampler = T.Resample(orig_freq=cfg.sample_rate, new_freq=16000) if cfg.sample_rate != 16000 else None
        del whisper_model
        gc.collect()
        return encoder, extractor, resampler
    except Exception as exc:
        log(f"  ⚠️ SR-FD validation disabled ({exc})")
        return None


@torch.no_grad()
def compute_srfd(bundle, real_audio, gen_audio):
    encoder, extractor, resampler = bundle
    try:
        device = next(encoder.parameters()).device
        dtype = next(encoder.parameters()).dtype
        min_a = min(real_audio.size(-1), gen_audio.size(-1))
        real_audio = real_audio[..., :min_a].float()
        gen_audio = gen_audio[..., :min_a].float()
        if resampler is not None:
            resamp = resampler.to(device)
            real_audio = resamp(real_audio)
            gen_audio = resamp(gen_audio)
        real_mel = extractor(
            real_audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=dtype)
        gen_mel = extractor(
            gen_audio.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device, dtype=dtype)
        real_feat = encoder(real_mel).last_hidden_state
        gen_feat = encoder(gen_mel).last_hidden_state
        return F.mse_loss(gen_feat.float().mean(dim=1), real_feat.float().mean(dim=1)).item()
    except Exception as exc:
        log(f"  ⚠️ SR-FD computation skipped: {exc}")
        return 0.0


def decode_tokens(token_ids):
    try:
        from preprocess.g2g import TAMIL_G2G_TOKENS
        vocab = TAMIL_G2G_TOKENS
        return " ".join(
            vocab[int(t)] if 0 <= int(t) < len(vocab) else "?"
            for t in token_ids
        )
    except Exception:
        return "(token decode unavailable)"


@torch.no_grad()
def synthesize_samples(net, ema, val_loader, device, cfg, step):
    """Rank-0 only: render 3 validation samples using EMA weights, then restore training weights."""
    try:
        batch = next(iter(val_loader))
    except Exception:
        return
    if batch is None:
        return

    samples_dir = os.path.join(cfg.checkpoint_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    n = min(3, batch["tokens"].size(0))
    tokens = batch["tokens"][:n].to(device)
    token_lens = batch["token_lens"][:n].to(device)
    mel = batch["mel"][:n].to(device)
    mel_lens = batch["mel_lens"][:n].to(device)
    ref_mel = torch.roll(mel, shifts=1, dims=0)
    ref_mel_lens = torch.roll(mel_lens, shifts=1)

    ema.store_backup(net)
    ema.copy_to(net)
    net.eval()
    lines = []
    try:
        with amp_context(device, cfg):
            for i in range(n):
                cur_tlen = int(token_lens[i].item())
                cur_tok = tokens[i:i+1, :cur_tlen]
                cur_tlens = token_lens[i:i+1]
                cur_mlen = int(ref_mel_lens[i].item())
                cur_ref = ref_mel[i:i+1, :, :cur_mlen]
                cur_reflens = ref_mel_lens[i:i+1]

                out = net(
                    cur_tok, cur_tlens,
                    mel=None, mel_lens=None,
                    gt_dur=None,
                    ref_mel=cur_ref, ref_mel_lens=cur_reflens,
                    return_audio=True,
                )
                gen_audio = out.get("gen_audio") if isinstance(out, dict) else None
                if gen_audio is not None:
                    wav = gen_audio[0].float().detach().cpu().numpy().astype(np.float32)
                    peak = np.max(np.abs(wav))
                    if peak > 1e-4:
                        wav = wav / max(peak, 1.0) * 0.95
                    fname = f"step_{step:07d}_sample_{i}.wav"
                    sf.write(os.path.join(samples_dir, fname), wav, cfg.sample_rate)
                    clean_toks = cur_tok[0].tolist()
                    lines.append(f"{fname}\t{decode_tokens(clean_toks)}")
    finally:
        net.train()
        ema.restore_backup(net)

    if lines:
        with open(os.path.join(samples_dir, f"step_{step:07d}_texts.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log(f"  🔊 Saved {len(lines)} EMA validation samples -> {samples_dir}")


@torch.no_grad()
def evaluate(model, val_loader, device, cfg, srfd_bundle=None, local_rank=0):
    net = unwrap_model(model)
    was_training = net.training
    net.eval()
    mel_fn = MelLoss(coarse_w=cfg.weight_mel_coarse, refined_w=cfg.weight_mel_refined,
                     sc_w=getattr(cfg, "weight_sc", 1.0),
                     lowband_w=getattr(cfg, "weight_mel_lowband", 1.0))

    total_mel = 0.0
    total_srfd = 0.0
    count = 0

    pbar = tqdm(val_loader, desc="Validating", unit="batch", ncols=120, disable=not is_main_process(local_rank))
    for batch in pbar:
        if batch is None:
            continue
        batch = to_device_batch(batch, device)
        ref_mel = torch.roll(batch["mel"], shifts=1, dims=0)
        ref_mel_lens = torch.roll(batch["mel_lens"], shifts=1)

        # Only rank 0 holds the Whisper SR-FD bundle; other ranks skip audio rendering.
        want_audio = srfd_bundle is not None and is_main_process(local_rank)
        with amp_context(device, cfg):
            out = net(
                batch["tokens"], batch["token_lens"],
                mel=batch["mel"], mel_lens=batch["mel_lens"],
                gt_dur=batch["gt_dur"] if cfg.use_gt_durations else None,
                ref_mel=ref_mel, ref_mel_lens=ref_mel_lens,
                return_audio=want_audio,
            )
        loss_mel, _, _ = mel_fn(
            out["mel_pred"].float(), out["mel_coarse"].float(),
            batch["mel"], batch["mel_lens"],
        )

        srfd_val = 0.0
        if srfd_bundle is not None and local_rank == 0 and out.get("gen_audio") is not None:
            srfd_val = compute_srfd(srfd_bundle, batch["audio"], out["gen_audio"])

        total_mel += loss_mel.item()
        total_srfd += srfd_val
        count += 1

        if is_main_process(local_rank):
            pbar.set_postfix({"mel": f"{loss_mel.item():.4f}", "srfd": f"{srfd_val:.4f}"})

    net.train(was_training)
    return total_mel / max(count, 1), total_srfd / max(count, 1)


def maybe_validate_and_save(model, net, ema, opt_g, sched_g, opt_d, val_loader,
                            device, cfg, global_step, best_val_loss,
                            srfd_bundle=None, local_rank=0, is_distributed=False):
    """Single validation + checkpointing entry point (industry-standard cadence).

    Called ONLY on step-based cadence (global_step % save_every == 0) or once at
    training end. Validates on all ranks (DDP requires collective participation),
    but checkpoint writes / sample synthesis / best-model logic are rank-0-only.

    Returns updated best_val_loss.
    """
    val_mel, val_srfd = evaluate(model, val_loader, device, cfg, srfd_bundle, local_rank)
    if is_main_process(local_rank):
        log(f"\n[Step {global_step}] Val Mel Loss: {val_mel:.4f} | Val SR-FD: {val_srfd:.4f}")
        extra = {"ema_state_dict": ema.state_dict(), "val_loss": float(val_mel)}
        # Only two rolling checkpoints are kept: latest.pt (resume point) and
        # best.pt (best val mel). latest.pt carries everything needed to resume
        # exactly: model + generator optimizer/scheduler + discriminator
        # optimizer + step + EMA shadow + val loss.
        save_checkpoint(
            os.path.join(cfg.checkpoint_dir, "latest.pt"),
            model, opt_g, sched_g, opt_d, global_step, extra,
        )
        if val_mel < best_val_loss:
            best_val_loss = val_mel
            save_checkpoint(
                os.path.join(cfg.checkpoint_dir, "best.pt"),
                model, opt_g, sched_g, opt_d, global_step, extra,
            )
            log(f"  🏆 New Best Model Saved (Mel Loss: {best_val_loss:.4f})")
        synthesize_samples(net, ema, val_loader, device, cfg, global_step)
    if is_distributed:
        dist.barrier()
    return best_val_loss


def train_worker(local_rank, world_size, cfg):
    is_distributed = world_size > 1
    if is_distributed:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=local_rank,
                world_size=world_size,
                device_id=device,
            )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log("=" * 60, local_rank)
    log("  TamilTTSv2 Training Pipeline (FastPitch Variant + FiLM Style + Staged GAN)", local_rank)
    log("=" * 60, local_rank)

    model = TamilTTSv2(cfg).to(device)

    if getattr(cfg, "vocoder_ckpt", None) and os.path.exists(cfg.vocoder_ckpt):
        log(f"  Loading Frozen Vocoder weights: {cfg.vocoder_ckpt}", local_rank)
        try:
            blob = torch.load(cfg.vocoder_ckpt, map_location="cpu", weights_only=False)
            state = blob.get("generator", blob) if isinstance(blob, dict) and hasattr(blob, "get") else blob
            model.vocoder.load_state_dict(state, strict=False)
        except Exception as exc:
            log(f"  ⚠️ Vocoder checkpoint not loaded ({exc}); using model-bundled init.", local_rank)
    for p in model.vocoder.parameters():
        p.requires_grad = False

    total_p, train_p = count_parameters(model)
    log(f"  Total Parameters    : {total_p / 1e6:.2f}M", local_rank)
    log(f"  Trainable Parameters: {train_p / 1e6:.2f}M", local_rank)

    mpd = MultiPeriodDiscriminator().to(device)
    mrd = MultiResolutionDiscriminator().to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        # Discriminators are NOT DDP-wrapped: the GAN loop backprops through them
        # twice per step (D update, then the generator's adversarial loss). Under
        # DDP that fires a second, rank-desynced round of gradient all-reduces on
        # D and deadlocks NCCL (watchdog kill at stage1_steps). Instead we keep
        # raw modules and manually average their grads across ranks before
        # opt_d.step() — same math as HiFi-GAN's DataParallel recipe.
        log(f"  🔥 DDP Active       : {world_size} GPUs (Rank {local_rank} on cuda:{local_rank}) "
            f"| discriminators: manual grad-averaging", local_rank)
    else:
        log(f"  🚀 Device Active    : {device}", local_rank)

    net = unwrap_model(model)

    def average_disc_grads():
        """All-reduce discriminator grads so every rank steps identical D weights."""
        if not is_distributed:
            return
        grads = [p.grad for p in disc_params if p.grad is not None]
        if not grads:
            return
        coalesced = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(coalesced, op=dist.ReduceOp.SUM)
        coalesced /= world_size
        for buf, grad in zip(torch._utils._unflatten_dense_tensors(coalesced, grads), grads):
            grad.copy_(buf)

    acoustic_params = [p for p in model.parameters() if p.requires_grad]
    disc_params = list(mpd.parameters()) + list(mrd.parameters())

    opt_g = torch.optim.AdamW(
        acoustic_params, lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8,
    )
    sched_g = get_lr_scheduler(opt_g, cfg.warmup_steps, cfg.total_steps)
    opt_d = torch.optim.AdamW(disc_params, lr=cfg.disc_lr, betas=(0.8, 0.99), eps=1e-8)

    mel_loss_fn = MelLoss(coarse_w=cfg.weight_mel_coarse, refined_w=cfg.weight_mel_refined,
                          sc_w=getattr(cfg, "weight_sc", 1.0),
                          lowband_w=getattr(cfg, "weight_mel_lowband", 1.0))
    dur_loss_fn = DurationLoss()
    pe_loss_fn = PitchEnergyLoss()
    disc_loss_fn = DiscriminatorLoss()
    gen_adv_fn = GeneratorAdversarialLoss()
    fm_loss_fn = FeatureMatchingLoss()

    ema = EMA(decay=cfg.ema_decay)
    ema.register(net)

    # SR-FD is a rank-0-only validation metric; loading Whisper on every DDP rank
    # wastes ~1.5 GB VRAM x world_size and slows startup.
    srfd_bundle = build_srfd_bundle(device, cfg) if is_main_process(local_rank) else None
    slm_cache = {}

    log(f"  Dataset             : {cfg.dataset_dir}", local_rank)
    train_ds, val_ds = build_tamil_datasets(cfg.dataset_dir, cfg)

    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False) if is_distributed else None
    use_workers = cfg.num_workers > 0
    # Row-group-local batches: one ~160 MB parquet group decode serves a whole
    # batch (vs. per-sample re-decode with shuffled access — the dominant cost).
    # DDP: rank/world_size shard the composed batches into disjoint sets.
    train_batch_sampler = RowGroupBatchSampler(
        train_ds, cfg.per_gpu_batch,
        rank=local_rank if is_distributed else 0,
        world_size=world_size if is_distributed else 1,
    )
    log(f"  RG-local batches    : {len(train_batch_sampler):,}/rank "
        f"({len(train_batch_sampler) * world_size:,} total)", local_rank)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
        collate_fn=tamil_tts_collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_workers,
        prefetch_factor=getattr(cfg, "prefetch_factor", 4) if use_workers else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.per_gpu_batch,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=tamil_tts_collate_fn,
        num_workers=min(2, cfg.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_workers,
        prefetch_factor=getattr(cfg, "prefetch_factor", 4) if use_workers else None,
        drop_last=False,
    )

    batches_per_gpu = len(train_loader)
    opt_steps_per_epoch = max(1, batches_per_gpu // cfg.grad_accum_steps)
    total_epochs = (cfg.total_steps // opt_steps_per_epoch) + 1

    log(f"  Batches/epoch       : {batches_per_gpu}", local_rank)
    log(f"  Opt steps/epoch     : {opt_steps_per_epoch}", local_rank)
    log(f"  Val batches         : {len(val_loader)}", local_rank)
    log(f"  Total target steps  : {cfg.total_steps}", local_rank)
    log(f"  Stage schedule      : GAN@{cfg.stage1_steps} | SLM@{cfg.slm_start_step}(+{cfg.slm_ramp_steps} ramp)", local_rank)
    log("=" * 60, local_rank)

    global_step = 0
    best_val_loss = float("inf")
    resume_path = getattr(cfg, "resume_path", None) or os.path.join(cfg.checkpoint_dir, "latest.pt")
    if resume_path and os.path.exists(resume_path):
        global_step = load_checkpoint(resume_path, model, opt_g, sched_g, opt_d)
        try:
            blob = torch.load(resume_path, map_location="cpu", weights_only=False)
            if isinstance(blob, dict) and blob.get("ema_state_dict"):
                ema.load_state_dict(blob["ema_state_dict"])
                log("  ✅ EMA shadow weights restored.", local_rank)
            if isinstance(blob, dict) and isinstance(blob.get("val_loss"), float):
                best_val_loss = blob["val_loss"]
        except Exception as exc:
            log(f"  ⚠️ Could not restore EMA/best-val metadata ({exc}).", local_rank)

    gc.collect()
    torch.cuda.empty_cache()

    try:
        model.train()
        mpd.train()
        mrd.train()

        for epoch in range(total_epochs):
            if global_step >= cfg.total_steps:
                break


            pbar = tqdm(
                enumerate(train_loader), total=batches_per_gpu,
                desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch", ncols=150,
                disable=not is_main_process(local_rank),
            )

            for batch_idx, batch in pbar:
                if batch is None or global_step >= cfg.total_steps:
                    continue

                batch = to_device_batch(batch, device)
                ref_mel = torch.roll(batch["mel"], shifts=1, dims=0)
                ref_mel_lens = torch.roll(batch["mel_lens"], shifts=1)

                gan_active = global_step >= cfg.stage1_steps
                cur_slm_w = slm_weight(global_step, cfg)

                with amp_context(device, cfg):
                    # The vocoder is NOT run on full utterances during training:
                    # make_gan_segments vocodes a random 32-frame window per
                    # sample instead (constant ~8k samples of audio per batch,
                    # vs. up to 10s x B through 256x upsampling that OOMed an
                    # L4 at the first GAN step).
                    out = model(
                        batch["tokens"], batch["token_lens"],
                        mel=batch["mel"], mel_lens=batch["mel_lens"],
                        gt_dur=batch["gt_dur"] if cfg.use_gt_durations else None,
                        gt_logf0=batch["log_f0"], voiced=batch["voiced"],
                        gt_energy=batch["energy"],
                        ref_mel=ref_mel, ref_mel_lens=ref_mel_lens,
                        style_dropout=cfg.style_dropout_p,
                        return_audio=False,
                    )

                    loss_mel, l_ref, l_crs = mel_loss_fn(
                        out["mel_pred"].float(), out["mel_coarse"].float(),
                        batch["mel"], batch["mel_lens"],
                    )
                    loss_dur = dur_loss_fn(out["log_dur"].float(), batch["gt_dur"], batch["token_lens"])
                    loss_f0, loss_energy = pe_loss_fn(
                        out["log_f0"], out["energy"],
                        batch["log_f0"], batch["voiced"], batch["energy"],
                        batch["mel_lens"],
                    )

                    g_loss = (
                        loss_mel
                        + cfg.weight_dur * loss_dur
                        + cfg.weight_f0 * loss_f0
                        + cfg.weight_energy * loss_energy
                    )

                    d_loss_val = adv_val = fm_val = 0.0

                    if gan_active:
                        net_vocoder = net.vocoder
                        real_cut, fake_cut = make_gan_segments(
                            net_vocoder, out["mel_pred"].float(), batch["mel_lens"],
                            batch["audio"], hop_length=cfg.hop_length,
                        )

                        scores_real, feats_real = run_discriminators(mpd, mrd, real_cut)
                        scores_fake_d, _ = run_discriminators(mpd, mrd, fake_cut.detach())
                        d_loss = disc_loss_fn(scores_real, scores_fake_d)

                        opt_d.zero_grad(set_to_none=True)
                        d_loss.backward()
                        average_disc_grads()  # ranks must agree before ANY of them steps
                        d_norm = torch.nn.utils.clip_grad_norm_(disc_params, cfg.max_grad_norm)
                        if torch.isfinite(d_norm):
                            opt_d.step()
                        else:
                            log(f"⚠️ [Step {global_step}] Non-finite D grads. Skipped D step.", local_rank)
                        opt_d.zero_grad(set_to_none=True)
                        d_loss_val = float(d_loss.item())

                        scores_fake_g, feats_fake_g = run_discriminators(mpd, mrd, fake_cut)
                        loss_adv = gen_adv_fn(scores_fake_g)
                        loss_fm = fm_loss_fn(feats_fake_g, feats_real)
                        g_loss = g_loss + cfg.weight_adv * loss_adv + cfg.weight_fm * loss_fm
                        adv_val = float(loss_adv.item())
                        fm_val = float(loss_fm.item())

                        if cur_slm_w > 0.0:
                            slm_fn = build_slm(device, cfg, slm_cache)
                            if slm_fn is not None:
                                loss_slm = slm_fn(real_cut, fake_cut)
                                g_loss = g_loss + cur_slm_w * loss_slm

                    (g_loss / cfg.grad_accum_steps).backward()

                is_accumulating = ((batch_idx + 1) % cfg.grad_accum_steps != 0)
                if not is_accumulating:
                    g_norm = torch.nn.utils.clip_grad_norm_(acoustic_params, cfg.max_grad_norm)
                    if torch.isfinite(g_norm):
                        opt_g.step()
                        sched_g.step()
                        ema.update(net)
                        global_step += 1
                    else:
                        log(f"⚠️ [Step {global_step}] Non-finite G grads. Skipping optimizer step.", local_rank)
                    opt_g.zero_grad(set_to_none=True)

                    if is_main_process(local_rank):
                        stage_tag = "GAN+SLM" if cur_slm_w > 0.0 else ("GAN" if gan_active else "REG")
                        postfix_dict = {
                            "step": global_step,
                            "stage": stage_tag,
                            "loss": f"{float(g_loss.item()):.2f}",
                            "mel": f"{float(l_ref.item()):.3f}",
                            "dur": f"{float(loss_dur.item()):.3f}",
                            "f0": f"{float(loss_f0.item()):.3f}",
                            "eng": f"{float(loss_energy.item()):.3f}",
                            "lr": f"{sched_g.get_last_lr()[0]:.1e}",
                        }
                        if gan_active:
                            postfix_dict["d"] = f"{d_loss_val:.3f}"
                            postfix_dict["adv"] = f"{adv_val:.3f}"
                        pbar.set_postfix(postfix_dict)

                    if global_step > 0 and global_step % cfg.save_every == 0:
                        best_val_loss = maybe_validate_and_save(
                            model, net, ema, opt_g, sched_g, opt_d, val_loader,
                            device, cfg, global_step, best_val_loss,
                            srfd_bundle=srfd_bundle, local_rank=local_rank,
                            is_distributed=is_distributed,
                        )

            # End of epoch: NO extra validation pass. Checkpointing/validation follow
            # the single step-based cadence (every save_every steps); an additional
            # epoch-end evaluate() doubled cost with no benefit.
            # A final save happens after the training loop below.

        # Final checkpoint + validation at end of training.
        best_val_loss = maybe_validate_and_save(
            model, net, ema, opt_g, sched_g, opt_d, val_loader,
            device, cfg, max(global_step, 1), best_val_loss,
            srfd_bundle=srfd_bundle, local_rank=local_rank,
            is_distributed=is_distributed,
        )

    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="TamilTTSv2 Training Pipeline (Staged FastPitch + FiLM + GAN)")
    parser.add_argument("--dataset_dir", nargs="+", default=None, help="Dataset directories")
    parser.add_argument("--vocoder_ckpt", type=str, default=None, help="Path to pre-trained frozen HiFi-GAN generator.pt")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Output directory for checkpoints/samples")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-GPU batch size override")
    parser.add_argument("--lr", type=float, default=None, help="Generator learning rate override")
    parser.add_argument("--disc_lr", type=float, default=None, help="Discriminator learning rate override")
    parser.add_argument("--steps", type=int, default=None, help="Total training steps override")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers per rank")
    args = parser.parse_args()

    cfg = Config()
    if args.resume:
        cfg.resume_path = args.resume
    if args.dataset_dir:
        cfg.dataset_dir = args.dataset_dir
    if args.vocoder_ckpt:
        cfg.vocoder_ckpt = args.vocoder_ckpt
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.batch_size:
        cfg.per_gpu_batch = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.disc_lr:
        cfg.disc_lr = args.disc_lr
    if args.steps:
        cfg.total_steps = args.steps
    if args.num_workers:
        cfg.num_workers = args.num_workers

    train_mode, num_devices = auto_configure_hardware(cfg)

    if train_mode == "ddp":
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        print(f"🔥 Spawning DDP multi-process training across {num_devices} GPUs...")
        torch.multiprocessing.spawn(
            train_worker,
            args=(num_devices, cfg),
            nprocs=num_devices,
            join=True,
        )
    else:
        train_worker(0, 1, cfg)


if __name__ == "__main__":
    main()
