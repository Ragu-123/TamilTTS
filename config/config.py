"""TamilTTS Configuration — Locked 68.55M architecture."""

class Config:
    # --- Paths ---
    dataset_dir    = "/kaggle/input/datasets/ragunathravi/shrutilipi-tamil/tamil"
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

    # --- Training ---
    total_steps  = 300_000
    batch_size   = 32
    num_workers  = 4
    learning_rate = 1e-4
    warmup_steps = 10_000
    save_every   = 5000
    val_every    = 2000
    log_every    = 10
    val_split    = 0.05

    # --- Loss Weights ---
    weight_mel   = 45.0
    weight_slm   = 1.0
    weight_dur   = 1.0
