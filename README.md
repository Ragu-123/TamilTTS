# TamilTTS v3

68.55M parameter Tamil TTS with Length Regulation (FastSpeech-style).

## Key Features (v3)
- **Length Regulation**: Duration predictor expands text features to match mel length
- **Corrupted Audio Handling**: Automatically skips bad samples (retries up to 5x)
- **Proper Mel Alignment**: mel_pred and mel_target have identical shapes
- **Multi-GPU**: Auto-detects and uses DataParallel
- **Checkpointing**: latest.pt, best.pt, step_N.pt, final.pt
- **Validation**: Every 2000 steps with IndicWhisper SR-FD metric

## Architecture (68.55M)
| Component | Params | Purpose |
|---|---|---|
| Text Encoder | 31.59M | 10-layer Transformer |
| Style Encoder | 4.54M | Conv+GRU reference style |
| Duration Predictor | 0.59M | Per-phoneme frame count |
| **Length Regulation** | 0 | Expands text to mel length |
| Diffusion Prosody | 14.30M | 8-block style-conditioned ResNet |
| Vocoder | 17.50M | HiFi-GAN V1 |
