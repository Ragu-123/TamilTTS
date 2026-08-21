"""TamilTTS Configuration — SOTA Decoupled Acoustic + Frozen HiFi-GAN Pipeline."""

class Config:
    # --- Paths (Supports single path or list of paths for multi-dataset combination) ---
    dataset_dir = [
        "/kaggle/input/datasets/ragunathravi/ai4bharat-indicvoices-r-tamil",
        "/kaggle/input/datasets/ragunathravi/ai4bharat-rasa-tamil",
    ]
    wavlm_dir      = "/kaggle/input/models/ragunathravi/wavlm-base-plus/pytorch/default/1"
    whisper_dir    = "/kaggle/input/notebooks/ragunathravi/tamil-asr/indicwhisper_tamil/tamil_models/whisper-medium-ta_alldata_multigpu"
    checkpoint_dir = "./checkpoints"

    # --- Pre-trained Universal HiFi-GAN Vocoder (Frozen) ---
    vocoder_ckpt   = "/kaggle/input/notebooks/sanjaynn/tamiltts-vocoder/indic_tts_tamil_clean/hifigan_generator.pt"
    vocoder_config = "/kaggle/input/notebooks/sanjaynn/tamiltts-vocoder/indic_tts_tamil_clean/hifigan_config.json"

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

    # --- Training ---
    total_steps      = 100_000
    per_gpu_batch    = 8        # Per-GPU batch size (auto-overridden by hardware detection)
    grad_accum_steps = 4        # Accumulate micro-batches (auto-overridden by hardware detection)
    num_workers      = 4
    learning_rate    = 1e-3     # Kokoro standard: AdamW with 1e-3 peak LR
    warmup_steps     = 4_000    # Linear warmup over ~4000 steps
    save_every       = 2_000

    # --- Loss Weights (Kokoro-82M Exact Standard) ---
    weight_mel_refined = 1.0     # PostNet refined mel loss
    weight_mel_coarse  = 0.5     # Pre-PostNet coarse mel loss
    weight_dur         = 0.02    # Log-scale duration loss (Kokoro uses 0.02)
    weight_slm         = 1.0     # Perceptual WavLM speech language model critic
