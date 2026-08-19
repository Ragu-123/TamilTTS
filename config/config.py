"""
TamilTTS Configuration — All hyperparameters and paths in one place.
Based on the locked 68.55M parameter architecture.
"""

class Config:
    # --- Paths (Kaggle defaults, override via CLI) ---
    dataset_dir = "/kaggle/input/datasets/ragunathravi/shrutilipi-tamil/tamil"
    wavlm_dir = "/kaggle/input/models/ragunathravi/wavlm-base-plus/pytorch/default/1"
    whisper_dir = "/kaggle/input/notebooks/ragunathravi/tamil-asr/indicwhisper_tamil/tamil_models/whisper-medium-ta_alldata_multigpu"
    checkpoint_dir = "./checkpoints"

    # --- Model Architecture (Locked) ---
    vocab_size = 128
    hidden_dim = 512
    text_encoder_layers = 10
    text_encoder_heads = 8
    style_dim = 256
    mel_channels = 80
    diffusion_blocks = 8
    duration_filter_channels = 256
    vocoder_initial_channels = 1024

    # --- Training ---
    total_steps = 300_000        # 300k steps for Stage 1 (390 hours)
    batch_size = 8               # Per GPU
    num_workers = 4
    learning_rate = 1e-4         # Generator
    disc_learning_rate = 2e-4    # Discriminator / Critics
    warmup_steps = 10_000        # Linear warmup
    save_every = 5000            # Checkpoint every N steps
    log_every = 10               # Print metrics every N steps
    max_audio_len = 48000        # 3 seconds at 16kHz
    max_text_len = 200           # Max phoneme tokens

    # --- Loss Weights ---
    weight_slm = 1.0
    weight_srfd = 1.0
    weight_mel = 45.0            # Mel L1 (dominant early on)
    weight_dur = 1.0             # Duration Huber
