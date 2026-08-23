# TamilTTSv2

FastPitch-variant Tamil TTS with FiLM style conditioning and staged GAN/SLM training (Kokoro / StyleTTS2 inspired), driving a frozen 22.05 kHz HiFi-GAN vocoder.

**Branch note:** `main` holds the legacy V1 stack (RAD-TTS learned aligner + diffusion). This branch, `feature/tamiltts-v2`, is the current V2 stack with code at the repo root (standard layout). The legacy V1 stack (RAD-TTS aligner + diffusion) lives untouched on `main`.

## Architecture

```
                 Tamil text
                     │  tokens_with_sil (G2G aksharas + MFA-style sil boundaries)
                     ▼
              tokens [B,Tt]
                     │
                     ▼
        ┌────────────────────────┐        ref mel [B,80,Tm]  (train: torch.roll(batch,1,dim=0))
        │ TextEncoder (6×pre-LN) │◄──────────────┐
        └───────────┬────────────┘               │
                    │ h                          ▼
                    ├────────────► DurationHead ──► log_dur / dur_pred
                    │                                    │ length regulator (GT dur in train)
                    │                                    ▼
                    │           PitchHead ──► log_f0 ──► PitchEmbedder ─┐
                    │           EnergyHead ─► energy ─► EnergyEmbedder ─┤
                    │                                                   ▼
                    └──────────►│   Decoder (4×FFT + FiLM(style))  ◄────┘
                                └───────────┬──────────────────────┘
                                            │
                        StyleEncoder(ref mel)│  (or trained default_style)
                                            ▼
                                     mel_proj ──► mel_coarse ──(+ PostNet)──► mel_pred [B,Tm,80]
                                            │
                             FullVocoder (frozen HiFi-GAN 22.05 kHz, 13.93M)
                                            ▼
                                      gen_audio [B,Ta]
```

Key wiring facts (all verified by `tests/preflight.py` on GPU):
- **Mel recipe = the vocoder's exact training recipe** (Coqui v0.0.13: ln-mel, center=True STFT, slaney basis, NO normalization). Round-trip envelope corr 0.91 / centroid corr 0.96.
- **Inference tokens match training**: `tokens_with_sil()` emits leading/trailing/pause `sil`, mirroring MFA transcription.
- **GT prosody mixing**: training feeds GT f0/energy; model mixes GT vs predicted conditioning 50/50 per step so the decoder sees inference-time distributions.
- **Energy conditions synthesis** via `EnergyEmbedder` (FastSpeech2-style), not just a loss output.
- **GAN/SLM losses see zero padding**: per-sample matched random crops (`crop_audio_pairs`).

## Components

| Path | Contents |
|---|---|
| `preprocess/g2g.py` | G2G akshara vocab (338/384 incl. decomposed vowel signs), `segment_tamil_g2g`, canonical `tokens_with_sil` |
| `scripts/generate_mfa_durations.py` | Produces the MFA durations `.pt` from TextGrids |
| `scripts/mfa_align_v1_reference.py` | V1 MFA alignment module (preserved reference) |
| `data/dataset.py` | Streaming parquet dataset, strict MFA filtering, dynamic collation |
| `data/audio_features.py` | `MelExtractor` (exact vocoder recipe), F0/energy extraction |
| `models/tamil_tts_v2.py` | The acoustic model + frozen vocoder |
| `models/discriminators.py` | MPD + MRD (HiFi-GAN style) |
| `losses/` | masked L1 suite, LSGAN D/G, feature matching, WavLM SLM |
| `utils/` | EMA, vocoder-free checkpoints, warmup+cosine LR |
| `train.py` | Staged trainer: spawn-DDP, bf16, step-based validate/save cadence |
| `inference.py` | CLI synthesis with optional reference-audio style |
| `tests/preflight.py` | 4 gates: param budget, grad flow, teacher-forcing fit, vocoder round-trip |
| `overfit_quality.py` | N-sample overfit quality harness |

## Parameter budget

| Block | Params |
|---|---|
| Trainable acoustic model | **~38.2 M** |
| Frozen HiFi-GAN vocoder | 13.9 M |
| **Total** | **~52.1 M** (cap 80M) |

## Staged training schedule

| Stage | Steps | Objectives |
|---|---|---|
| 1 — Regression | 0 → 25k | dual-mel L1, log-duration L1, voiced F0 L1, energy L1 |
| 2 — +GAN | ≥ 25k | LSGAN adv + feature matching (MPD/MRD); D-step before G-step on cropped audio |
| 3 — +SLM | ≥ 40k (+10k ramp) | WavLM hidden-state L1 (layers 3/7/11 @16 kHz), final weight 0.1 |

Validation/checkpoint cadence is **step-based only** (`save_every=2000`): `latest.pt`, `best.pt`, `step_N.pt` + EMA validation wavs under `checkpoints/samples/`. No epoch-end double validation. Checkpoints exclude the frozen vocoder (~56 MB saved per file).

Epoch math: 69k samples ÷ effective batch 64 ≈ 1082 steps/epoch → 150k steps ≈ 139 epochs (in the standard band: StyleTTS2 300 ep @460h, Kokoro <20 ep @<100h curated, FastPitch 1000 ep @24h).

## Training on Kaggle 4×L4

```bash
git clone --branch feature/tamiltts-v2 --single-branch https://github.com/Ragu-123/TamilTTS.git
cd TamilTTS
python train.py \
    --dataset_dir /kaggle/input/datasets/ragunathravi/ai4bharat-indicvoices-r-tamil \
                  /kaggle/input/datasets/ragunathravi/ai4bharat-rasa-tamil \
    --vocoder_ckpt /kaggle/input/notebooks/sanjaynn/tamiltts-vocoder/indic_tts_tamil_clean/hifigan_generator.pt \
    --checkpoint_dir /kaggle/working/checkpoints \
    --steps 150000
```

Resume after a session cap:

```bash
python train.py ... --resume /kaggle/working/checkpoints/latest.pt
```

Other overrides: `--batch_size`, `--lr`, `--disc_lr`, `--num_workers`.

## Inference

```bash
# default voice (trained default_style)
python inference.py --text "வணக்கம், தமிழ் உரைப்பெடுப்பு செயல்முறை" --out hello.wav

# clone style from a reference clip
python inference.py --text "இது ஒரு சோதனை வாக்கியம்." \
    --ref_audio samples/ref.wav --checkpoint checkpoints/best.pt --out test.wav
```

EMA weights are preferred automatically when present in the checkpoint.

## Verification before long runs

```bash
python tests/smoke_test.py                       # CPU-only: losses, EMA, checkpoint round-trip
python tests/preflight.py --steps 400 --wav X.wav # GPU: all 4 gates must PASS
python overfit_quality.py --steps 4000           # 20-sample overfit; TF≈FR means durations converged
```
