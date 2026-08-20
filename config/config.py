"""TamilTTS Configuration — 68.55M End-to-End Pipeline."""

class Config:
    # --- Paths (Supports single path or list of paths for multi-dataset combination) ---
    dataset_dir = [
        "/kaggle/input/datasets/ragunathravi/ai4bharat-indicvoices-r-tamil",
        "/kaggle/input/datasets/ragunathravi/ai4bharat-rasa-tamil",
    ]
    wavlm_dir      = "/kaggle/input/models/ragunathravi/wavlm-base-plus/pytorch/default/1"
    whisper_dir    = "/kaggle/input/notebooks/ragunathravi/tamil-asr/indicwhisper_tamil/tamil_models/whisper-medium-ta_alldata_multigpu"
    checkpoint_dir = "./checkpoints"

    # --- Model Architecture ---
    vocab_size               = 256      # Covers all Tamil Unicode (0x0B80-0x0BFF) + digits + punctuation
    hidden_dim               = 512
    text_encoder_layers      = 10
    text_encoder_heads       = 8
    style_dim                = 256
    mel_channels             = 80
    diffusion_blocks         = 8
    duration_filter_channels = 256
    vocoder_initial_channels = 512

    # --- Audio ---
    sample_rate   = 16000
    n_fft         = 1024
    hop_length    = 256
    max_audio_len = 48000   # 3 seconds @ 16kHz
    max_text_len  = 200
    max_mel_len   = 188     # max_audio_len / hop_length

    # --- Training (DDP + Gradient Accumulation) ---
    total_steps      = 100_000
    per_gpu_batch    = 8        # Per-GPU batch size (fits comfortably in 10-16 GB VRAM)
    grad_accum_steps = 4        # Accumulate 4 micro-batches
    # Effective batch = per_gpu_batch(8) * num_gpus(4) * grad_accum(4) = 128
    num_workers      = 4
    learning_rate    = 1e-4
    warmup_steps     = 2_000
    save_every       = 2_000

    # --- Loss Weights ---
    weight_mel = 45.0           # Standard acoustic mel reconstruction weight (HiFi-GAN / StyleTTS 2)
    weight_slm = 1.0            # Perceptual WavLM speech language model critic
    weight_dur = 1.0            # Duration predictor alignment loss
