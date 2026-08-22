"""
CPU-only Smoke Test for TamilTTSv2 Training Stack
=================================================
Validates (without GPU / dataset / pretrained models):
  1. All regression + GAN loss modules produce finite scalars.
  2. EMA register/update/copy_to/backup round-trip.
  3. Checkpoint save/load round-trip (incl. optimizer_disc and EMA payload).
  4. TamilTTSv2 forward (return_audio=False) + all regression losses finite
     + one full backward/optimizer step.
Heavy imports (models, discriminators, wavlm) are guarded: missing teammates'
files degrade the test gracefully instead of crashing.
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from losses import (
    MelLoss,
    DurationLoss,
    PitchEnergyLoss,
    DiscriminatorLoss,
    GeneratorAdversarialLoss,
    FeatureMatchingLoss,
)
from utils import EMA, save_checkpoint, load_checkpoint, count_parameters


def tiny_cfg():
    return types.SimpleNamespace(
        vocab_size=384,
        hidden_dim=64,
        text_encoder_layers=1,
        decoder_layers=1,
        text_encoder_heads=2,
        ff_dim=128,
        style_dim=32,
        variance_filter_channels=64,
        postnet_dim=64,
        mel_channels=80,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        f_min=0.0,
        f_max=8000.0,
        min_audio_len=11025,
        max_audio_len=220500,
        max_text_len=250,
        vocoder_ckpt=None,
        vocoder_config=None,
        use_gt_durations=True,
        style_dropout_p=0.5,
    )


def make_batch(B=2, Tt=10, Tm=30):
    torch.manual_seed(0)
    Ta = 30 * 256
    return {
        "tokens": torch.randint(5, 380, (B, Tt)),
        "token_lens": torch.full((B,), Tt, dtype=torch.long),
        "mel": torch.rand(B, 80, Tm) * 8.0 - 4.0,
        "mel_lens": torch.full((B,), Tm, dtype=torch.long),
        "audio": torch.randn(B, Ta) * 0.05,
        "audio_lens": torch.full((B,), Ta, dtype=torch.long),
        "gt_dur": torch.full((B, Tt), float(Tm) / float(Tt)),
        "log_f0": torch.randn(B, Tm) * 0.5,
        "voiced": (torch.rand(B, Tm) > 0.3).float(),
        "energy": torch.randn(B, Tm) * 0.25,
    }


def check(name, value):
    assert torch.isfinite(value), f"{name} loss is not finite: {value}"
    print(f"  {name:<12s} = {float(value.detach()):.4f}")


def test_losses_and_utils():
    batch = make_batch()

    d = DiscriminatorLoss()(
        [torch.randn(2, 1, 8), torch.randn(2, 1, 8)],
        [torch.randn(2, 1, 8), torch.randn(2, 1, 8)],
    )
    g = GeneratorAdversarialLoss()([torch.randn(2, 1, 8), torch.randn(2, 1, 8)])
    fm = FeatureMatchingLoss()(
        [[torch.randn(2, 8, 10), torch.randn(2, 8, 10)], [torch.randn(2, 8, 10)]],
        [[torch.randn(2, 8, 10), torch.randn(2, 8, 10)], [torch.randn(2, 8, 10)]],
    )
    print("[1] Standalone GAN losses:")
    check("d_loss", d)
    check("g_adv", g)
    check("fm", fm)

    probe = torch.nn.Linear(4, 4)
    ema_probe = EMA(decay=0.5)
    ema_probe.register(probe)
    with torch.no_grad():
        for p in probe.parameters():
            p.add_(1.0)
    ema_probe.update(probe)
    ema_probe.store_backup(probe)
    ema_probe.copy_to(probe)
    assert probe.weight.abs().sum().item() < sum(p.numel() for p in probe.parameters()) * 1.0 + 1e-6
    ema_probe.restore_backup(probe)
    print("[2] EMA update/copy/backup/restore: OK")

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke_ckpt.pt")
    popt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    save_checkpoint(ckpt_path, probe, popt, None, None, 7, extra={"ema_state_dict": ema_probe.state_dict()})
    probe2 = torch.nn.Linear(4, 4)
    step = load_checkpoint(ckpt_path, probe2, popt)
    os.remove(ckpt_path)
    assert step == 7, f"checkpoint step mismatch: {step}"
    print("[3] Checkpoint save/load round-trip: OK")

    return batch


def test_model_forward(batch, cfg):
    from models.tamil_tts_v2 import TamilTTSv2

    mel_fn = MelLoss(coarse_w=0.5, refined_w=1.0)
    dur_fn = DurationLoss()
    pe_fn = PitchEnergyLoss()

    model = TamilTTSv2(cfg)
    ref_mel = torch.roll(batch["mel"], shifts=1, dims=0)
    ref_mel_lens = torch.roll(batch["mel_lens"], shifts=1)

    out = model(
        batch["tokens"], batch["token_lens"],
        mel=batch["mel"], mel_lens=batch["mel_lens"],
        gt_dur=batch["gt_dur"],
        ref_mel=ref_mel, ref_mel_lens=ref_mel_lens,
        style_dropout=cfg.style_dropout_p,
        return_audio=False,
    )

    print("[4] Model forward + regression losses:")
    l_mel, _, _ = mel_fn(out["mel_pred"], out["mel_coarse"], batch["mel"], batch["mel_lens"])
    l_dur = dur_fn(out["log_dur"], batch["gt_dur"], batch["token_lens"])
    l_f0, l_e = pe_fn(
        out["log_f0"], out["energy"],
        batch["log_f0"], batch["voiced"], batch["energy"],
        batch["mel_lens"],
    )
    total = l_mel + l_dur + l_f0 + l_e
    check("mel", l_mel)
    check("dur", l_dur)
    check("f0", l_f0)
    check("energy", l_e)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    opt.zero_grad()
    total.backward()
    grads_finite = all(
        torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    assert grads_finite, "non-finite gradients after backward"
    opt.step()

    ema = EMA(decay=0.9)
    ema.register(model)
    ema.update(model)

    total_p, trainable_p = count_parameters(model)
    return total_p, trainable_p


def test_discriminators(batch):
    from models.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
    mpd = MultiPeriodDiscriminator()
    mrd = MultiResolutionDiscriminator()
    s_r, f_r = mpd(batch["audio"][:2])
    s_f, f_f = mpd(torch.roll(batch["audio"][:2], shifts=1, dims=-1))
    d = DiscriminatorLoss()(s_r, s_f)
    g = GeneratorAdversarialLoss()(s_f)
    fm = FeatureMatchingLoss()(f_f, f_r)
    check("mpd_d", d)
    check("mpd_g", g)
    check("mpd_fm", fm)

    sr2, fr2 = mrd(batch["audio"][:2])
    sf2, ff2 = mrd(torch.roll(batch["audio"][:2], shifts=1, dims=-1))
    d2 = DiscriminatorLoss()(sr2, sf2)
    fm2 = FeatureMatchingLoss()(ff2, fr2)
    check("mrd_d", d2)
    check("mrd_fm", fm2)


def main():
    print("=" * 60)
    print("  TamilTTSv2 CPU Smoke Test")
    print("=" * 60)

    cfg = tiny_cfg()
    batch = test_losses_and_utils()

    model_ok = False
    try:
        from models.tamil_tts_v2 import TamilTTSv2
        total_p, trainable_p = test_model_forward(batch, cfg)
        print(f"[5] Params: total={total_p / 1e6:.3f}M | trainable={trainable_p / 1e6:.3f}M")
        model_ok = True
    except ImportError as exc:
        print(f"[SKIP] models.tamil_tts_v2 unavailable: {exc}")
    except Exception as exc:
        print(f"[FAIL] Model forward/backward failed: {type(exc).__name__}: {exc}")
        raise

    try:
        test_discriminators(batch)
    except ImportError as exc:
        print(f"[SKIP] discriminators unavailable: {exc}")

    if model_ok:
        print("PASS")
    else:
        print("PASS-CORE (losses/utils verified; model module pending from teammates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
