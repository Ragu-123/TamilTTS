"""TamilTTS Configuration — Locked 68.55M architecture."""

class Config:
    # --- Paths (Supports comma-separated datasets for multi-dataset combination) ---
    dataset_dir    = "/kaggle/input/datasets/ragunathravi/ai4bharat-indicvoices-r-tamil,/kaggle/input/datasets/ragunathravi/ai4bharat-rasa-tamil"
    wavlm_dir      = "/kaggle/input/models/ragunathravi/wavlm-base-plus/pytorch/default/1"
    whisper_dir    = "/kaggle/input/notebooks/ragunathravi/tamil-asr/indicwhisper_tamil/tamil_models/whisper-medium-ta_alldata_multigpu"
    checkpoint_dir = "./checkpoints"

    # --- Model Architecture (Locked @ 68.55M) ---
    vocab_size               = 128
    hidden_dim               = 512
    text_encoder_layers      = 10
    text_encoder_heads       = 8
    style_dim                = 256
    mel_channels             = 80
    diffusion_blocks         = 8
    duration_filter_channels = 256
    vocoder_initial_channels = 1024

    # --- Audio ---
    sample_rate  = 16000
    n_fft        = 1024
    hop_length   = 256
    max_audio_len = 48000   # 3 seconds @ 16kHz
    max_text_len  = 200
    max_mel_len   = 188     # max_audio_len / hop_length

    # --- Training (DDP + Gradient Accumulation) ---
    total_steps      = 300_000
    per_gpu_batch    = 8        # Per-GPU batch size (fits in ~10 GB VRAM)
    grad_accum_steps = 4        # Accumulate 4 micro-batches before optimizer step
    # Effective batch = per_gpu_batch(8) * num_gpus(4) * grad_accum(4) = 128
    num_workers    = 4
    learning_rate  = 1e-4
    warmup_steps   = 10_000
    save_every     = 5000
    val_every      = 2000
    log_every      = 10
    val_split      = 0.05

    # --- Loss Weights ---
    weight_mel   = 45.0
    weight_slm   = 1.0
    weight_dur   = 1.0
