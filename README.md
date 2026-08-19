# TamilTTS

68.55M parameter Text-to-Speech engine for Tamil. Modified StyleTTS 2 architecture.

## Architecture (Locked)
| Component | Params | Details |
|---|---|---|
| Text Encoder | 31.59M | 10-layer Transformer, 512-dim, 8 heads |
| Style Encoder | 4.54M | Conv1d + GRU, 256-dim style vector |
| Duration Predictor | 0.59M | Conv1d + LayerNorm |
| Diffusion Prosody | 14.30M | 8-block ResNet, style-conditioned |
| Vocoder | 17.50M | HiFi-GAN V1 (4x upsampling) |
| **Total** | **68.55M** | |

## Critics (Frozen, validation only)
| Critic | Purpose | Speed |
|---|---|---|
| WavLM Base Plus | SLM adversarial loss (emotion/prosody) | Training loss |
| AI4Bharat IndicWhisper | SR-FD metric (Tamil phonetic quality) | Validation metric |

## Dataset
- **Shrutilipi Tamil**: 281,508 utterances (~390 hours)
- **Split**: 95% train (267,432) / 5% validation (14,076)

## Training Config
- **Steps**: 300,000 (Stage 1)
- **Optimizer**: Lion, lr=1e-4
- **Scheduler**: Cosine Annealing + 10k warmup
- **Checkpoint**: Every 5,000 steps + best model tracking
- **Validation**: Every 2,000 steps (Mel L1 + SR-FD)

## Project Structure
```
TamilTTS/
├── config/config.py           # Hyperparameters
├── data/dataset.py            # Shrutilipi loader + train/val split
├── models/
│   ├── text_encoder.py
│   ├── style_encoder.py
│   ├── duration_predictor.py
│   ├── diffusion.py
│   ├── vocoder.py
│   └── tamil_tts.py
├── losses/losses.py           # SLM + SR-FD
├── utils/utils.py             # Checkpointing, LR scheduler
└── train.py                   # Main training loop
```

## Run
```bash
python train.py \
  --dataset_dir /path/to/shrutilipi/tamil \
  --wavlm_dir /path/to/wavlm-base-plus \
  --whisper_dir /path/to/indicwhisper-tamil
```
