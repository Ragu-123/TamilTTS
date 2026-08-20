# TamilTTS v4 (DDP)

68.55M parameter Tamil TTS with DistributedDataParallel multi-GPU training.

## Key Features (v4)
- **DDP Multi-GPU**: All GPUs at ~95-100% utilization (no GPU 0 bottleneck)
- **Gradient Accumulation**: Effective batch 128 without OOM (8 per GPU × 4 GPUs × 4 accum)
- **Length Regulation**: Duration predictor expands text features to match mel length
- **indic-nlp Normalizer**: Unicode normalization before character tokenization
- **Corrupted Audio Handling**: Automatically skips bad samples to next index
- **Checkpointing**: latest.pt, best.pt, step_N.pt, final.pt
- **Validation**: Every 2000 steps with IndicWhisper SR-FD metric

## How to Launch

### Multi-GPU (4x L4 / 4x T4) — Recommended
```bash
torchrun --nproc_per_node=4 train.py \
    --dataset_dir /kaggle/input/datasets/ragunathravi/shrutilipi-tamil/tamil \
    --wavlm_dir /kaggle/input/models/ragunathravi/wavlm-base-plus/pytorch/default/1 \
    --whisper_dir /kaggle/input/notebooks/ragunathravi/tamil-asr/indicwhisper_tamil/tamil_models/whisper-medium-ta_alldata_multigpu
```

### Single GPU Fallback
```bash
python train.py \
    --dataset_dir /kaggle/input/datasets/ragunathravi/shrutilipi-tamil/tamil \
    --wavlm_dir /kaggle/input/models/ragunathravi/wavlm-base-plus/pytorch/default/1 \
    --whisper_dir /kaggle/input/notebooks/ragunathravi/tamil-asr/indicwhisper_tamil/tamil_models/whisper-medium-ta_alldata_multigpu
```

### Kaggle Notebook Cell
```python
import os
os.environ["DATASET"]  = "/kaggle/input/datasets/ragunathravi/shrutilipi-tamil/tamil"
os.environ["WAVLM"]    = "/kaggle/input/models/ragunathravi/wavlm-base-plus/pytorch/default/1"
os.environ["WHISPER"]   = "/kaggle/input/notebooks/ragunathravi/tamil-asr/indicwhisper_tamil/tamil_models/whisper-medium-ta_alldata_multigpu"

!torchrun --nproc_per_node=4 train.py \
    --dataset_dir $DATASET --wavlm_dir $WAVLM --whisper_dir $WHISPER
```

## Architecture (68.55M)
| Component | Params | Purpose |
|---|---|---|
| Text Encoder | 31.59M | 10-layer Transformer |
| Style Encoder | 4.54M | Conv+GRU reference style |
| Duration Predictor | 0.59M | Per-phoneme frame count |
| Length Regulation | 0 | Expands text to mel length |
| Diffusion Prosody | 14.30M | 8-block style-conditioned ResNet |
| Vocoder | 17.50M | HiFi-GAN V1 |

## GPU Memory Layout (DDP)
| GPU | TamilTTS | WavLM | IndicWhisper | Activations | Total |
|---|---|---|---|---|---|
| GPU 0 | 0.27 GB | 0.36 GB | 0.6 GB | ~8 GB | ~9.2 GB |
| GPU 1 | 0.27 GB | 0.36 GB | 0.6 GB | ~8 GB | ~9.2 GB |
| GPU 2 | 0.27 GB | 0.36 GB | 0.6 GB | ~8 GB | ~9.2 GB |
| GPU 3 | 0.27 GB | 0.36 GB | 0.6 GB | ~8 GB | ~9.2 GB |
