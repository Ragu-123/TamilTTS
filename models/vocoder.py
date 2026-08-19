import torch
import torch.nn as nn

class ResBlock1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(torch.relu(self.conv1(x)))

class FullVocoder(nn.Module):
    """HiFi-GAN V1 scale vocoder — ~17.5M params."""
    def __init__(self, in_channels=80, upsample_initial_channel=1024):
        super().__init__()
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, padding=3)
        self.up1 = nn.ConvTranspose1d(1024, 512, 16, 8, padding=4)
        self.res1 = nn.Sequential(*[ResBlock1d(512) for _ in range(3)])
        self.up2 = nn.ConvTranspose1d(512, 256, 16, 8, padding=4)
        self.res2 = nn.Sequential(*[ResBlock1d(256) for _ in range(3)])
        self.up3 = nn.ConvTranspose1d(256, 128, 4, 2, padding=1)
        self.res3 = nn.Sequential(*[ResBlock1d(128) for _ in range(3)])
        self.up4 = nn.ConvTranspose1d(128, 64, 4, 2, padding=1)
        self.res4 = nn.Sequential(*[ResBlock1d(64) for _ in range(3)])
        self.conv_post = nn.Conv1d(64, 1, 7, 1, padding=3)

    def forward(self, x):
        x = self.conv_pre(x.transpose(1, 2))
        x = self.res1(torch.relu(self.up1(x)))
        x = self.res2(torch.relu(self.up2(x)))
        x = self.res3(torch.relu(self.up3(x)))
        x = self.res4(torch.relu(self.up4(x)))
        return self.conv_post(x).transpose(1, 2)
