"""
Dual-GPU 100-Sample 2,000-Step Tamil TTS Training Script (DDP Supported)
========================================================================
- Trains on 100 diverse Tamil sentences for 2,000 steps on Dual GPUs (2x T4).
- Evaluates on 5 unseen validation sentences at steps 500, 1000, 1500, 2000.
- Uses all latest fixes: Zero-init PostNet, -11.51 silence masking, and unpadded rendering.
"""
import os, sys, time, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import soundfile as sf
from tqdm import tqdm

from config.config import Config
from models.tamil_tts_v2 import TamilTTSv2
from losses.losses import MelLoss, DurationLoss, PitchEnergyLoss
from data.dataset import build_tamil_datasets, tamil_tts_collate_fn, build_tamil_vocab
from utils.utils import EMA, unwrap_model

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size, True
    elif torch.cuda.is_available():
        return 0, 0, torch.cuda.device_count(), False
    return 0, 0, 1, False

def main():
    rank, local_rank, world_size, is_dist = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    cfg = Config()
    cfg.learning_rate = 5e-4
    
    if rank == 0:
        print("=" * 75, flush=True)
        print(f"  TAMIL TTS: 100 SAMPLES FOR 2,000 STEPS (DUAL GPU - {world_size} GPUs)", flush=True)
        print("=" * 75, flush=True)
    
    train_ds, val_ds = build_tamil_datasets(cfg.dataset_dir, cfg)
    char2id, vocab_size = build_tamil_vocab()
    id2char = {v: k for k, v in char2id.items()}
    
    train_subset = [train_ds[i] for i in range(1, 101)]
    val_subset = [val_ds[i] for i in range(5)]
    
    if rank == 0:
        print(f"✓ Training on 100 diverse sentences | Validating on 5 unseen sentences.", flush=True)
    
    sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True) if is_dist else None
    train_loader = DataLoader(
        train_subset, batch_size=8, shuffle=(sampler is None),
        sampler=sampler, collate_fn=tamil_tts_collate_fn, num_workers=2, pin_memory=True
    )
    
    model = TamilTTSv2(cfg).to(device)
    if is_dist:
        net = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    elif world_size > 1 and torch.cuda.is_available():
        net = torch.nn.DataParallel(model)
    else:
        net = model
        
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
    ema = EMA(decay=0.999)
    if rank == 0:
        ema.register(model)
        
    mel_fn = MelLoss(coarse_w=0.5, refined_w=1.0, sc_w=1.0, lowband_w=2.0)
    dur_fn = DurationLoss()
    pe_fn = PitchEnergyLoss()
    
    out_dir = "/kaggle/working/results_100_samples_2000_steps"
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        
    epoch = 0
    step = 0
    t0 = time.time()
    pbar = tqdm(total=2000, desc="Dual-GPU Training", ncols=125, disable=(rank != 0))
    
    while step < 2000:
        if is_dist and sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        
        for batch in train_loader:
            step += 1
            if step > 2000:
                break
                
            for k, v in batch.items():
                if torch.is_tensor(v):
                    batch[k] = v.to(device)
                    
            ref_mel = torch.roll(batch["mel"], 1, 0)
            ref_mel_lens = torch.roll(batch["mel_lens"], 1)
            
            out = net(
                batch["tokens"], batch["token_lens"],
                mel=batch["mel"], mel_lens=batch["mel_lens"],
                gt_dur=batch["gt_dur"],
                gt_logf0=batch["log_f0"],
                voiced=batch["voiced"],
                gt_energy=batch["energy"],
                ref_mel=ref_mel,
                ref_mel_lens=ref_mel_lens,
                return_audio=False
            )
            
            l_m, l_r, l_c = mel_fn(out["mel_pred"], out["mel_coarse"], batch["mel"], batch["mel_lens"])
            l_d = dur_fn(out["log_dur"], batch["gt_dur"], batch["token_lens"])
            l_p, l_e = pe_fn(out["log_f0"], out["energy"], batch["log_f0"], batch["voiced"], batch["energy"], batch["mel_lens"])
            
            loss = l_m + l_d + l_p + l_e
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if rank == 0:
                ema.update(unwrap_model(net))
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.3f}",
                    "mel": f"{l_r.item():.3f}",
                    "dur": f"{l_d.item():.3f}"
                })
                
            # Milestone evaluation
            if step in [500, 1000, 1500, 2000] and rank == 0:
                core_model = unwrap_model(net)
                ema.store_backup(core_model)
                ema.copy_to(core_model)
                core_model.eval()
                
                with torch.no_grad():
                    for val_i in range(len(val_subset)):
                        v_item = val_subset[val_i]
                        v_toks = v_item[0][:v_item[1]].unsqueeze(0).to(device)
                        v_tlens = torch.tensor([v_item[1]], device=device)
                        v_rmel = v_item[2][:, :v_item[3]].unsqueeze(0).to(device)
                        v_mlens = torch.tensor([v_item[3]], device=device)
                        
                        fr_out = core_model(v_toks, v_tlens, mel=None, mel_lens=None, gt_dur=None, ref_mel=v_rmel, ref_mel_lens=v_mlens, return_audio=True)
                        synth_wav = fr_out["gen_audio"][0].cpu().numpy()
                        
                        mel_p = fr_out["mel_pred"][0].cpu().numpy().T
                        f_vel = float(np.mean(np.abs(np.diff(mel_p, axis=1))))
                        rms = float(np.sqrt(np.mean(synth_wav**2)))
                        
                        fname = f"step_{step}_unseen_val_{val_i}.wav"
                        sf.write(os.path.join(out_dir, fname), synth_wav, cfg.sample_rate)
                        tok_chars = [id2char.get(tid, '?') for tid in v_toks[0].tolist()]
                        print(f"\\n  [Step {step} | Val Sentence {val_i}] -> {fname} | Duration: {len(synth_wav)/22050:.2f}s | RMS: {rms:.4f} | Formants: {f_vel:.3f}", flush=True)
                        print(f"    Text: {' '.join(tok_chars[:12])}...", flush=True)
                        
                ckpt_path = os.path.join(out_dir, f"model_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": core_model.state_dict(),
                    "opt_state_dict": opt.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                }, ckpt_path)
                print(f"  💾 Saved checkpoint -> {ckpt_path}", flush=True)
                
                ema.restore_backup(core_model)
                core_model.train()

    if rank == 0:
        pbar.close()
        print("\\n" + "=" * 75, flush=True)
        print(f"✓ Dual-GPU Training Complete in {time.time()-t0:.1f}s!", flush=True)
        print(f"Checkpoints & Audios saved to: {out_dir}", flush=True)
        print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
