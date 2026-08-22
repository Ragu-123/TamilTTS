# TamilTTSv2

FastPitch-variant Tamil TTS with FiLM style conditioning and staged GAN training (Kokoro / StyleTTS2 inspired), driving a frozen 22.05 kHz HiFi-GAN vocoder.

> **Note:** this stack lives on a **non-main branch**. Merge into main before release/deployment.

## Architecture

```
                 Tamil text
                     │  segment_tamil_g2g (G2G aksharas, vocab=384)
                     ▼
              tokens [B,Tt]
                     │
                     ▼
        ┌────────────────────────┐        ref mel [B,80,Tm]  (train: torch.roll(batch,1,dim=0))
        │ TextEncoder (6×FFT)    │◄──────────────┐
        └───────────┬────────────┘               │
                    │ h                          ▼
                    ├────────────► DurationPredictor ──► log_dur / dur_pred
                    │                                    │ length regulator (GT dur in train)
                    │                                    ▼
                    │           ┌──────── FiLM(style) ─────────────┐
                    └──────────►│   Decoder (4×FFT + FiLM)         │
                                └───────────┬──────────────────────┘
                                            │
                        StyleEncoder(ref mel)│
                                            ▼
                                     PostNet (256d) ──► mel_pred [B,Tm,80] (+ mel_coarse)
                                            │
                             variance adapters: log_f0, energy
                                            ▼
                            FullVocoder (frozen HiFi-GAN 22.05 kHz, 13.93M)
                                            ▼
                                      gen_audio [B,Ta]
```

## Components

| Path | Owner | Contents |
|---|---|---|
| `preprocess/g2g.py` | teammate | `TAMIL_G2G_TOKENS`, `VOCAB_SIZE=384`, `segment_tamil_g2g` |
| `data/dataset.py`, `data/audio_features.py` | teammate | dict-batch collate, dataset builder, `MelExtractor`, F0/energy extraction |
| `models/tamil_tts_v2.py` | teammate | `TamilTTSv2(cfg)` + frozen `.vocoder` (FullVocoder) |
| `models/discriminators.py` | teammate | `MultiPeriodDiscriminator`, `MultiResolutionDiscriminator` |
| `config/` | this branch | v1 Kaggle paths verbatim, architecture/training/stage hyper-parameters |
| `losses/` | this branch | masked L1 suite, LSGAN D/G losses, feature matching, WavLM `SLMLoss` |
| `utils/` | this branch | `EMA`, module-aware checkpoints, warmup+cosine LR scheduler |
| `train.py` | this branch | staged trainer (spawn-based DDP, bf16, two optimizers, EMA sampling) |
| `inference.py` | this branch | CLI synthesis with optional reference-audio style |

## Parameter budget (target)

| Block | Approx. params |
|---|---|
| Token embedding (384×512) | 0.20 M |
| Text encoder (6 × FFT, d=512, ff=1024) | ~12.6 M |
| Duration predictor (filter=256) | ~1.2 M |
| Pitch + energy predictors | ~2.0 M |
| Style encoder (→ style_dim 256) | ~3.5 M |
| Decoder (4 × FFT + FiLM) | ~8.8 M |
| PostNet (256d) + heads/projections | ~3.0 M |
| **Total trainable** | **~50 M** |
| HiFi-GAN vocoder (frozen) | 13.93 M |

Exact counts are printed at training launch (`count_parameters`).

## Staged training

| Stage | Steps | Objectives |
|---|---|---|
| 1 — Regression | 0 → 25k (`stage1_steps`) | weighted masked dual-mel (`mel_refined` 1.0, `mel_coarse` 0.5), log-duration L1, voiced-restricted F0 L1, energy L1 |
| 2 — +GAN | ≥ 25k | LSGAN adversarial + feature matching through MPD+MRD; discriminator step **before** generator step on detached audio; separate AdamW (disc_lr=1e-4, betas 0.8/0.99) |
| 3 — +SLM | ≥ 40k (`slm_start_step`) | WavLM hidden-state L1 (layers 3/7/11 @16 kHz), weight linearly ramped over 10k steps to 0.1 |

Extras: reference-condition roll (each utterance styled by a *different* utterance — anti style-leakage), GT-duration supervision, EMA (0.999) used for validation sample synthesis, bf16 autocast + TF32, non-finite grad skipping.

Checkpoints every 2000 steps on rank 0: `latest.pt`, `best.pt`, `step_N.pt` (all include `ema_state_dict`); validation wavs + decoded texts under `checkpoints/samples/`.

## Training on Kaggle 4xL4

Spawn-based DDP — just run:

```bash
python train.py
```

Useful overrides:

```bash
python train.py --resume ./checkpoints/latest.pt --steps 150000
python train.py --batch_size 12 --lr 1e-4 --disc_lr 1e-4
python train.py --checkpoint_dir /kaggle/working/checkpoints
```

## Inference

```bash
# default voice (checkpoint style prior)
python inference.py --text "வணக்கம், தமிழ் உரைப்பெடுப்பு செயல்முறை" --out hello.wav

# clone style from a reference clip (any sample rate, mono-mixed)
python inference.py --text "இது ஒரு சோதனை வாக்கியம்." \
    --ref_audio samples/ref.wav --checkpoint checkpoints/best.pt --out test.wav
```

EMA weights are preferred automatically when present in the checkpoint.

## Smoke test

```bash
python tests/smoke_test.py   # CPU-only: losses, EMA, checkpoint round-trip, tiny forward/backward
```
