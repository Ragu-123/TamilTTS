import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class PeriodDiscriminator(nn.Module):
    """HiFi-GAN period sub-discriminator (weight_norm Conv2d stack)."""
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.norm = weight_norm

        channels = [32, 128, 256, 512, 1024]
        in_ch = 1
        self.blocks = nn.ModuleList()
        for out_ch in channels:
            self.blocks.append(weight_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))
            ))
            in_ch = out_ch
        self.final1 = weight_norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0)))
        self.final2 = weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0)))

    def forward(self, x):
        """
        x: [B, T] audio
        Returns (score [B,1,T'], feat_list).
        """
        feat_list = []
        B, T = x.shape
        pad = (self.period - (T % self.period)) % self.period
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad), mode="reflect")
        x = x.view(B, 1, -1, self.period)  # [B, 1, T/p, p]

        for block in self.blocks:
            x = torch.nn.functional.leaky_relu(block(x), 0.1)
            feat_list.append(x)
        x = torch.nn.functional.leaky_relu(self.final1(x), 0.1)
        feat_list.append(x)
        x = self.final2(x)
        feat_list.append(x)
        return x, feat_list


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator over periods (2,3,5,7,11)."""
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, audio):
        """
        audio: [B, T]
        Returns (scores: list of [B,1,T'], feat_lists: list of list[Tensor]).
        """
        scores, feat_lists = [], []
        for disc in self.discriminators:
            score, feats = disc(audio)
            scores.append(score)
            feat_lists.append(feats)
        return scores, feat_lists


class ResDiscriminator(nn.Module):
    """HiFi-GAN multi-resolution sub-discriminator on STFT magnitude."""
    def __init__(self, resolution=(1024, 256, 1024)):
        super().__init__()
        n_fft, hop, win = resolution
        self.n_fft = n_fft
        self.hop = hop
        self.win = win

        channels = [32, 128, 256, 512, 1024]
        in_ch = 1
        self.convs = nn.ModuleList()
        # pool after layers 1 and 3 (indices), loosely following HiFi-GAN MRD
        self.pool_after = {0, 3}
        for i, out_ch in enumerate(channels):
            self.convs.append(weight_norm(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
            ))
            in_ch = out_ch
        self.final = weight_norm(nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, audio):
        """
        audio: [B, T]
        Returns (score, feat_list).
        """
        x = torch.stft(
            audio.float(),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.win,
            window=torch.hann_window(self.win, device=audio.device),
            center=True,
            return_complex=True,
        )
        mag = x.abs().unsqueeze(1)          # [B, 1, F, T']
        x = torch.log(mag + 1e-7)           # log magnitude

        feat_list = []
        for i, conv in enumerate(self.convs):
            x = torch.nn.functional.leaky_relu(conv(x), 0.1)
            feat_list.append(x)
            if i in self.pool_after:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        score = self.final(x)
        feat_list.append(score)
        return score, feat_list


class MultiResolutionDiscriminator(nn.Module):
    """Multi-resolution discriminator over three STFT resolutions."""
    def __init__(self, resolutions=((1024, 256, 1024), (2048, 512, 2048), (512, 128, 512))):
        super().__init__()
        self.discriminators = nn.ModuleList([ResDiscriminator(r) for r in resolutions])

    def forward(self, audio):
        """
        audio: [B, T]
        Returns (scores list, feat_lists list).
        """
        scores, feat_lists = [], []
        for disc in self.discriminators:
            score, feats = disc(audio)
            scores.append(score)
            feat_lists.append(feats)
        return scores, feat_lists
