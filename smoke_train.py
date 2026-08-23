"""
Fast production-path smoke run: exercises the FULL train.py pipeline (DDP multi-GPU,
REG -> GAN -> SLM staging, EMA, checkpointing, SR-FD validation, sample synthesis)
in ~300 optimizer steps instead of 150k.

Usage (Kaggle T4 x2):  python smoke_train.py
Everything here mirrors train.py exactly -- only the schedule is compressed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import Config

# --- Compressed schedule (mechanics identical to full run) ---
Config.total_steps      = 300
Config.warmup_steps     = 30
Config.stage1_steps     = 100   # GAN active from step 100
Config.slm_start_step   = 200   # SLM ramp 200 -> 250
Config.slm_ramp_steps   = 50
Config.save_every       = 150   # validate + ckpt + EMA samples at 150 & 300
Config.per_gpu_batch    = 4
Config.grad_accum_steps = 1
Config.num_workers      = 2
Config.prefetch_factor  = 2
Config.style_dropout_p  = 0.0   # validated: lower mel floor than 0.5
Config.checkpoint_dir   = "/kaggle/working/ttsv2_smoke"
Config.resume_path      = None

import train

if __name__ == "__main__":
    train.main()
