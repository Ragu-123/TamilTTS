import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1(nn.Module):
    """
    HiFi-GAN Multi-Receptive Field (MRF) Residual Block.
    Uses multiple dilated convolutions with LeakyReLU to capture wide harmonic context.
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size,
                stride=1, dilation=d, padding=((kernel_size - 1) * d) // 2
            )
            for d in dilation
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(
                channels, channels, kernel_size,
                stride=1, dilation=1, padding=(kernel_size - 1) // 2
            )
            for _ in dilation
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x


class FullVocoder(nn.Module):
    """
    HiFi-GAN V1 scale Neural Vocoder.
    Converts 80-channel mel spectrograms -> 16kHz continuous audio waveforms.
    Upsampling: 8 * 8 * 2 * 2 = 256x (exact match for hop_length=256).
    """
    def __init__(self, in_channels=80, upsample_initial_channel=512):
        super().__init__()
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)

        # Upsampling layers: 8 * 8 * 2 * 2 = 256
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(upsample_initial_channel, 256, 16, 8, padding=4),
            nn.ConvTranspose1d(256, 128, 16, 8, padding=4),
            nn.ConvTranspose1d(128, 64, 4, 2, padding=1),
            nn.ConvTranspose1d(64, 32, 4, 2, padding=1),
        ])

        # Multi-Receptive Field (MRF) ResBlocks for each stage
        # Combines kernel sizes [3, 7, 11] for rich harmonic reproduction
        self.resblocks = nn.ModuleList([
            nn.ModuleList([ResBlock1(256, k, (1, 3, 5)) for k in (3, 7, 11)]),
            nn.ModuleList([ResBlock1(128, k, (1, 3, 5)) for k in (3, 7, 11)]),
            nn.ModuleList([ResBlock1(64, k, (1, 3, 5)) for k in (3, 7, 11)]),
            nn.ModuleList([ResBlock1(32, k, (1, 3, 5)) for k in (3, 7, 11)]),
        ])

        self.conv_post = nn.Conv1d(32, 1, 7, 1, padding=3)

    def forward(self, x):
        """
        x: [B, T_mel, 80] or [B, 80, T_mel]
        Returns: [B, T_audio] in range [-1.0, 1.0]
        """
        if x.dim() == 3 and x.size(2) == 80:
            x = x.transpose(1, 2)  # [B, 80, T_mel]

        x = self.conv_pre(x)
        for up, res_list in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = 0
            for res in res_list:
                xs = xs + res(x)
            x = xs / len(res_list)

        x = F.leaky_relu(x, 0.1)
        x = torch.tanh(self.conv_post(x))
        return x.squeeze(1)  # [B, T_audio]
