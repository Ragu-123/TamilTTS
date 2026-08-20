# TamilTTS v5.2 (Combined Studio Dataset — ~76k Samples)

68.55M parameter Tamil TTS trained on clean studio-recorded speech with DistributedDataParallel.

## Key Features (v5.2)
- **Combined High-Fidelity Dataset (~76k samples)**: Automatically merges **AI4Bharat Rasa** (Expressive Studio TTS) and **AI4Bharat IndicVoices-R** (Studio Read Speech) using PyTorch `ConcatDataset`.
- **Zero Radio Background Noise**: 100% studio-quality condenser microphone speech.
- **Auto-DDP Multi-GPU**: Auto-detects 4 GPUs across `python train.py` and `torchrun`.
- **Gradient Accumulation**: Effective batch size 128 (8 per GPU × 4 GPUs × 4 accum) with ~9.5 GB VRAM per GPU.
- **Length Regulation**: Duration predictor expands text features to match mel length.
- **indic-nlp Normalizer**: Unicode normalization before character tokenization.
- **Checkpointing**: `latest.pt`, `best.pt`, `step_N.pt`, `final.pt`.
- **Validation**: Every 2000 steps with AI4Bharat IndicWhisper SR-FD metric.

## Combined Datasets
1. **AI4Bharat IndicVoices-R (Tamil Clean Read Speech)**:
   - Kaggle Path: `/kaggle/input/datasets/ragunathravi/ai4bharat-indicvoices-r-tamil`
   - Files: `train-*.parquet` (~39.3k samples) + `test-*.parquet` (~293 samples)
2. **AI4Bharat Rasa (Tamil Studio Expressive TTS)**:
   - Kaggle Path: `/kaggle/input/datasets/ragunathravi/ai4bharat-rasa-tamil`
   - Files: `train.parquet` (33,005 samples) + `test.parquet` (3,656 samples)

**Total Combined Studio Training Set**: **~76,254 samples (~120+ hours of studio speech)**

## How to Launch on Kaggle

### 1. Launch with BOTH Datasets Combined (Default):
```bash
!python train.py
```
*(Automatically detects and combines both datasets into ~76k samples)*

### 2. Launch with a Single Specific Dataset:
```bash
# Only IndicVoices-R (~39.6k samples)
!python train.py --dataset_dir /kaggle/input/datasets/ragunathravi/ai4bharat-indicvoices-r-tamil

# Only Rasa (~36.7k samples)
!python train.py --dataset_dir /kaggle/input/datasets/ragunathravi/ai4bharat-rasa-tamil
```

### 3. Launch with `torchrun`:
```bash
!torchrun --nproc_per_node=4 train.py
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
