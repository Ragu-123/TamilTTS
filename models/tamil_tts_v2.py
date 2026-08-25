import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import sequence_mask, length_regulate
    from .text_encoder import TextEncoder
    from .style_encoder import StyleEncoder
    from .predictors import DurationHead, PitchHead, EnergyHead, PitchEmbedder, EnergyEmbedder
    from .decoder import MelDecoder
    from .postnet import PostNet
    from .vocoder import load_pretrained_vocoder
except ImportError:
    from modules import sequence_mask, length_regulate
    from text_encoder import TextEncoder
    from style_encoder import StyleEncoder
    from predictors import DurationHead, PitchHead, EnergyHead, PitchEmbedder, EnergyEmbedder
    from decoder import MelDecoder
    from postnet import PostNet
    from vocoder import load_pretrained_vocoder


class TamilTTSv2(nn.Module):
    """
    TamilTTS v2: FastPitch-style acoustic model with FiLM style conditioning.
    Text encoder -> duration/pitch/energy heads -> FiLM-conditioned mel decoder
    -> postnet refinement -> frozen HiFi-GAN vocoder.
    """
    def __init__(self, cfg):
        super().__init__()
        hidden_dim = getattr(cfg, "hidden_dim", 512)
        text_encoder_layers = getattr(cfg, "text_encoder_layers", 6)
        decoder_layers = getattr(cfg, "decoder_layers", 4)
        num_heads = getattr(cfg, "heads", None) or getattr(cfg, "text_encoder_heads", 8)
        ff_dim = getattr(cfg, "ff_dim", 1024)
        postnet_dim = getattr(cfg, "postnet_dim", 256)
        style_dim = getattr(cfg, "style_dim", 256)
        filter_channels = getattr(cfg, "variance_filter_channels", 256)
        self.mel_channels = getattr(cfg, "mel_channels", 80)
        vocab_size = getattr(cfg, "vocab_size", 384)

        self.style_dim = style_dim

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, hidden_dim=hidden_dim,
            num_layers=text_encoder_layers, num_heads=num_heads,
            ff_dim=ff_dim,
        )
        self.style_encoder = StyleEncoder(
            mel_channels=self.mel_channels, hidden_dim=hidden_dim, style_dim=style_dim
        )
        self.default_style = nn.Parameter(torch.randn(1, style_dim) * 0.02)

        self.duration_head = DurationHead(hidden_dim, filter_channels)
        self.pitch_head = PitchHead(hidden_dim, filter_channels)
        self.energy_head = EnergyHead(hidden_dim, filter_channels)
        self.pitch_embedder = PitchEmbedder(hidden_dim)
        self.energy_embedder = EnergyEmbedder(hidden_dim)

        self.mel_decoder = MelDecoder(
            hidden_dim=hidden_dim, num_layers=decoder_layers, num_heads=num_heads,
            ff_dim=ff_dim, style_dim=style_dim,
        )
        self.mel_proj = nn.Linear(hidden_dim, self.mel_channels)
        # Bias init at the mean of the target mel range so early predictions are in-distribution.
        # Raw Coqui ln-mel targets average around -7 (range ~[-14, 2]); the old -2.5 assumed
        # the previous [-4, 4] normalized range.
        with torch.no_grad():
            self.mel_proj.bias.fill_(-7.0)

        self.postnet = PostNet(mel_dim=self.mel_channels, postnet_dim=postnet_dim)

        self.vocoder = load_pretrained_vocoder(
            device=("cuda" if torch.cuda.is_available() else "cpu"),
            checkpoint_path=getattr(cfg, "vocoder_ckpt", None)
        )

    def forward(self, tokens, token_lens, mel=None, mel_lens=None, gt_dur=None,
                gt_logf0=None, voiced=None, gt_energy=None, ref_mel=None,
                ref_mel_lens=None, style_dropout=0.0, return_audio=False,
                target_len=None):
        """
        tokens:   [B, Tt] int token ids
        token_lens: [B]
        Returns dict of predictions.
        """
        B, Tt = tokens.shape
        device = tokens.device

        # 1. Text mask (True = pad)
        text_mask = ~sequence_mask(token_lens, Tt)  # [B, Tt]

        # 2. Style vector
        if ref_mel is not None:
            style = self.style_encoder(ref_mel.detach(), ref_mel_lens)  # [B, S]
        else:
            style = self.default_style.expand(B, -1)
        if self.training and style_dropout > 0:
            drop = (torch.rand(B, 1, device=style.device) < style_dropout).float()
            default_expanded = self.default_style.expand(B, -1)
            style = drop * default_expanded + (1.0 - drop) * style

        # 3. Text encoding
        x = self.text_encoder(tokens, text_mask)  # [B, Tt, H]

        # 4. Duration prediction
        log_dur = self.duration_head(x, text_mask)                    # [B, Tt]
        dur_pred = torch.exp(log_dur).clamp(1.0, 100.0)
        dur_pred = dur_pred.masked_fill(text_mask, 0.0)

        # 5. Expansion (strictly 0 frames for padding tokens)
        if gt_dur is not None:
            durations = gt_dur.float().masked_fill(text_mask, 0.0)
        else:
            durations = torch.round(dur_pred).masked_fill(text_mask, 0.0)

        rounded = durations.long()
        out_lens = rounded.sum(dim=1).clamp(min=16)

        if gt_dur is not None and mel_lens is not None:
            # DataParallel splits the batch per-GPU; each replica would otherwise
            # compute its own max length and outputs could not be gathered.
            mel_len_target = max(16, int(target_len) if target_len is not None else int(mel_lens.max().item()))
            mel_mask = ~sequence_mask(mel_lens, mel_len_target)
        else:
            mel_len_target = max(16, int(target_len) if target_len is not None else int(out_lens.max().item()))
            mel_mask = ~sequence_mask(out_lens, mel_len_target)

        expanded = length_regulate(x, durations, mel_len_target)      # [B, Tm, H]

        # 6. Pitch conditioning
        pred_logf0 = self.pitch_head(expanded, mel_mask)              # [B, Tm]
        f0_for_embed = pred_logf0
        if self.training and gt_logf0 is not None:
            # Mix GT and predicted pitch so the decoder/pitch-embedder also see
            # the predicted-f0 distribution they receive at inference.
            # GT-only conditioning makes synthesis collapse (robotic/garbled).
            use_gt = torch.rand(()).item() < 0.5
            f0_for_embed = gt_logf0 if use_gt else pred_logf0.detach()
        # Some dataset clips carry f0 frame counts that differ from mel length
        # (MFA rounding / edge cases). Align to the decoder time axis explicitly.
        if f0_for_embed.size(1) != expanded.size(1):
            f0_for_embed = F.interpolate(
                f0_for_embed.unsqueeze(1), size=expanded.size(1), mode="nearest"
            ).squeeze(1)
        dec_in = expanded + self.pitch_embedder(f0_for_embed)

        # 7. Energy: predicted on the pitch-conditioned stream (same wire as before),
        # then embedded back into the decoder input so it actually conditions
        # synthesis (FastSpeech2-style) instead of being a supervised-only output.
        energy_pred = self.energy_head(dec_in, mel_mask)              # [B, Tm]
        energy_for_embed = energy_pred
        if self.training and gt_energy is not None:
            # Same GT/pred mix policy as pitch, keyed to the SAME coin flip so the
            # decoder always sees one coherent prosody condition per step.
            if not self.training or use_gt:
                energy_for_embed = gt_energy
            else:
                energy_for_embed = energy_pred.detach()
        if energy_for_embed.size(1) != expanded.size(1):
            energy_for_embed = F.interpolate(
                energy_for_embed.unsqueeze(1), size=expanded.size(1), mode="nearest"
            ).squeeze(1)
        dec_in = dec_in + self.energy_embedder(energy_for_embed)

        # 8. Mel decoding

        # 8. Mel decoding
        h = self.mel_decoder(dec_in, style, mel_mask)                 # [B, Tm, H]
        mel_coarse = self.mel_proj(h)                                 # [B, Tm, 80]
        mel_pred = mel_coarse + self.postnet(mel_coarse)
        
        # Mask padding frames to acoustic silence (-11.5129) so padding never produces buzz
        mel_coarse = mel_coarse.masked_fill(mel_mask.unsqueeze(-1), -11.5129)
        mel_pred = mel_pred.masked_fill(mel_mask.unsqueeze(-1), -11.5129)

        # 9. Audio synthesis
        gen_audio = None
        if return_audio:
            gen_audio = self.vocoder(mel_pred)

        return {
            "mel_pred": mel_pred,
            "mel_coarse": mel_coarse,
            "log_dur": log_dur,
            "dur_pred": dur_pred,
            "log_f0": pred_logf0,
            "energy": energy_pred,
            "style": style,
            "gen_audio": gen_audio,
        }


if __name__ == "__main__":
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        hidden_dim=512,
        text_encoder_layers=6,
        decoder_layers=4,
        heads=8,
        ff_dim=1024,
        postnet_dim=256,
        style_dim=256,
        variance_filter_channels=256,
        mel_channels=80,
        vocab_size=384,
        vocoder_ckpt=None,
    )

    model = TamilTTSv2(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:    {total_params / 1e6:.2f}M")
    print(f"Trainable params:{trainable_params / 1e6:.2f}M")

    B, Tt, Tm = 2, 12, 40
    tokens = torch.randint(1, 384, (B, Tt))
    token_lens = torch.tensor([Tt, Tt])
    durs_per_row = [Tm // Tt] * (Tt - 1) + [Tm - (Tt - 1) * (Tm // Tt)]
    gt_dur = torch.tensor([durs_per_row for _ in range(B)], dtype=torch.float)
    gt_logf0 = torch.randn(B, Tm) * 2.0 + 5.0

    model.eval()
    with torch.no_grad():
        out = model(tokens, token_lens, gt_dur=gt_dur, gt_logf0=gt_logf0)

    assert out["mel_pred"].shape == (B, Tm, 80), out["mel_pred"].shape
    assert out["mel_coarse"].shape == (B, Tm, 80), out["mel_coarse"].shape
    assert out["log_dur"].shape == (B, Tt), out["log_dur"].shape
    assert out["dur_pred"].shape == (B, Tt), out["dur_pred"].shape
    assert out["log_f0"].shape == (B, Tm), out["log_f0"].shape
    assert out["energy"].shape == (B, Tm), out["energy"].shape
    assert out["style"].shape == (B, cfg.style_dim), out["style"].shape
    for k in ["mel_pred", "mel_coarse", "log_dur", "dur_pred", "log_f0", "energy"]:
        assert torch.isfinite(out[k]).all(), f"{k} has non-finite values"

    print("OK")
