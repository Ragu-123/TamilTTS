"""
TamilTTS Configuration — FastPitch + RAD-TTS Alignment + Frozen 22.05kHz HiFi-GAN Pipeline.
"""

class Config:
    # --- Paths ---
    dataset_dir = [
        "/kaggle/input/datasets/ragunathravi/ai4bharat-indicvoices-r-tamil",
        "/kaggle/input/datasets/ragunathravi/ai4bharat-rasa-tamil",
    ]
    durations_file = "/kaggle/input/datasets/sanjaynn/tamil-mfa-durations-76k/tamil_mfa_durations_76k.pt"
    wavlm_dir      = "/kaggle/input/models/ragunathravi/wavlm-base-plus/pytorch/default/1"
    whisper_dir    = "/kaggle/input/notebooks/ragunathravi/tamil-asr/indicwhisper_tamil/tamil_models/whisper-medium-ta_alldata_multigpu"
    checkpoint_dir = "./checkpoints"

    # --- Pre-trained Universal HiFi-GAN Vocoder (Frozen, 22.05 kHz) ---
    vocoder_ckpt   = "/kaggle/input/notebooks/sanjaynn/tamiltts-vocoder/indic_tts_tamil_clean/hifigan_generator.pt"
    vocoder_config = "/kaggle/input/notebooks/sanjaynn/tamiltts-vocoder/indic_tts_tamil_clean/hifigan_config.json"

    # --- Model Architecture ---
    vocab_size               = 384      # Covers all 273 Tamil G2G akshara units + sil + special tokens
    hidden_dim               = 512
    text_encoder_layers      = 10
    text_encoder_heads       = 8
    style_dim                = 256
    mel_channels             = 80
    diffusion_blocks         = 8
    duration_filter_channels = 256
    aligner_dim              = 128
    vocoder_initial_channels = 512

    # --- Audio (22.05 kHz Standard for HiFi-GAN V1) ---
    sample_rate   = 22050
    n_fft         = 1024
    hop_length    = 256
    f_min         = 0.0
    f_max         = 8000.0
    min_audio_len = 11025   # 0.5 seconds @ 22.05kHz
    max_audio_len = 220500  # 10.0 seconds @ 22.05kHz
    max_text_len  = 250

    # --- Training Parameters ---
    total_steps      = 100_000
    per_gpu_batch    = 32       # 32 samples per GPU (~12-13GB VRAM on L4)
    grad_accum_steps = 1        # Instant 1-step updates across 4 GPUs (Effective batch = 128)
    num_workers      = 4
    learning_rate    = 1.5e-4   # FastPitch / StyleTTS2 standard LR
    warmup_steps     = 2_000
    save_every       = 2_000

    # --- Loss Weights (Staged & Warmed-up Training) ---
    weight_mel_refined = 1.0     # PostNet refined mel loss (Masked L1)
    weight_mel_coarse  = 0.5     # Pre-PostNet coarse mel loss (Masked L1)
    weight_dur         = 1.0     # Duration MSE loss (1.0 for robust duration learning)
    weight_align       = 1.0     # Exact RAD-TTS Forward-Sum Loss
    weight_bin         = 1.0     # Exact RAD-TTS Binarization Loss target
    bin_warmup_steps   = 5_000   # Binarization loss warmup (0.0 -> 1.0 over 5k steps)
    weight_slm         = 0.0     # Stage 1: 0.0 (Stage 2 after step 10k: 0.1)
    use_gt_durations   = True    # Kokoro-style ground-truth duration supervision
