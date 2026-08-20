# TamilTTS v5 (Studio High-Fidelity Multi-GPU DDP)

68.55M parameter Tamil TTS trained on clean studio-recorded speech with DistributedDataParallel.

## Key Features (v5)
- **Universal Clean Dataset Support**: Native plug-and-play for **AI4Bharat Rasa** (Studio Expressive TTS) & **AI4Bharat IndicVoices-R** (Studio Read Speech).
- **Auto-DDP Multi-GPU**: Auto-detects 4 GPUs across `python train.py` and `torchrun`.
- **Gradient Accumulation**: Effective batch size 128 (8 per GPU × 4 GPUs × 4 accum) with ~9.5 GB VRAM per GPU.
- **Length Regulation**: Duration predictor expands text features to match mel length.
- **indic-nlp Normalizer**: Unicode normalization before character tokenization.
- **Checkpointing**: `latest.pt`, `best.pt`, `step_N.pt`, `final.pt`.
- **Validation**: Every 2000 steps with AI4Bharat IndicWhisper SR-FD metric.

## Datasets Supported
1. **AI4Bharat Rasa (Tamil Studio TTS)**:
   - Kaggle Path: `/kaggle/input/datasets/ragunathravi/ai4bharat-rasa-tamil`
   - Files: `train.parquet` (33,005 samples) + `test.parquet` (3,656 samples)
   - Features: `text`, `audio`, `gender`, `style`, `duration`
2. **AI4Bharat IndicVoices-R (Tamil Read Speech)**:
   - Kaggle Path: `/kaggle/input/datasets/ragunathravi/ai4bharat-indicvoices-r-tamil`
   - Features: `normalized`, `text`, `audio`, `speaker_id`
3. **AI4Bharat Shrutilipi (Tamil Broadcast)**:
   - Kaggle Path: `/kaggle/input/datasets/ragunathravi/shrutilipi-tamil/tamil`

## How to Launch on Kaggle

### With Default Rasa Tamil Studio Dataset:
```bash
!python train.py
```

### With IndicVoices-R Tamil:
```bash
!python train.py --dataset_dir /kaggle/input/datasets/ragunathravi/ai4bharat-indicvoices-r-tamil
```

### With `torchrun`:
```bash
!torchrun --nproc_per_node=4 train.py --dataset_dir /kaggle/input/datasets/ragunathravi/ai4bharat-rasa-tamil
```

## Architecture (68.55M)
| Component | Params | Purpose |
|---|---|---|
| Text Encoder | 31.59M | 10-layer Transformer |
| Style Encoder | 4.54M | Conv+GRU reference style extractor |
| Duration Predictor | 0.59M | Per-phoneme frame count predictor |
| Length Regulation | 0 | Dynamic feature expansion to mel length |
| Diffusion Prosody | 14.30M | 8-block style-conditioned ResNet |
| Vocoder | 17.50M | HiFi-GAN V1 16kHz neural vocoder |
