# TamilTTS

A 68.55M parameter Text-to-Speech engine for Tamil, based on Modified StyleTTS 2.

## Architecture
| Component | Parameters |
|---|---|
| Text Encoder (10-layer Transformer) | 31.59M |
| Style Encoder (Conv + GRU) | 4.54M |
| Duration Predictor | 0.59M |
| Diffusion Prosody (8-block ResNet) | 14.30M |
| Vocoder (HiFi-GAN V1) | 17.50M |
| **Total** | **68.55M** |

## Critics (Frozen, not trained)
- **WavLM** — Adversarial SLM loss for emotion/prosody
- **AI4Bharat IndicWhisper (Vistaar)** — SR-FD loss for Tamil phonetic accuracy

## Project Structure
```
TamilTTS/
├── config/
│   └── config.py        # All hyperparameters and paths
├── data/
│   └── dataset.py       # Shrutilipi Dataset + DataLoader
├── models/
│   ├── text_encoder.py
│   ├── style_encoder.py
│   ├── duration_predictor.py
│   ├── diffusion.py
│   ├── vocoder.py
│   └── tamil_tts.py     # Main model
├── losses/
│   └── losses.py        # SLM + SR-FD losses
├── utils/
│   └── utils.py         # Checkpointing, LR scheduler
└── train.py             # Main training script
```

## Training
```bash
python train.py \
  --dataset_dir /path/to/shrutilipi \
  --wavlm_dir /path/to/wavlm \
  --whisper_dir /path/to/indicwhisper
```

## Key Features
- Multi-GPU via `nn.DataParallel` (auto-detects)
- tqdm progress bars with live loss metrics
- Checkpoint saving every 5000 steps + best model tracking
- Resume training from any checkpoint
- Cosine Annealing LR with 10,000 step warmup
- 300,000 total training steps for Stage 1
