"""
TamilTTSv2 Configuration — FastPitch Variant + FiLM Style Conditioning + Staged GAN Training.
Paths and audio recipe carried over verbatim from TamilTTS v1 (frozen 22.05 kHz HiFi-GAN pipeline).
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
    vocab_size               = 384
    hidden_dim               = 512
    text_encoder_layers      = 6
    decoder_layers           = 4
    text_encoder_heads       = 8
    ff_dim                   = 1024
    style_dim                = 256
    variance_filter_channels = 256
    postnet_dim              = 256
    mel_channels             = 80

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
    total_steps      = 150_000
    per_gpu_batch    = 16
    grad_accum_steps = 1
    num_workers      = 8
    prefetch_factor  = 4
    learning_rate    = 1.5e-4
    disc_lr          = 1e-4
    weight_decay     = 0.01
    warmup_steps     = 3_000
    save_every       = 2_000
    max_grad_norm    = 1.0
    ema_decay        = 0.999
    use_bf16         = True
    resume_path      = None

    # --- Loss Weights ---
    weight_mel_refined = 1.0
    weight_mel_coarse  = 0.5
    weight_dur         = 1.0
    weight_f0          = 1.0
    weight_energy      = 1.0
    weight_adv         = 1.0
    weight_fm          = 1.0
    weight_slm_final   = 0.1

    # --- Staged Training Schedule ---
    stage1_steps     = 25_000
    slm_start_step   = 40_000
    slm_ramp_steps   = 10_000
    use_gt_durations = True
    style_dropout_p  = 0.5
