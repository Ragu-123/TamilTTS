"""
PostNet: Convolutional Network for Mel-Spectrogram Refinement
============================================================
Based on Tacotron 2 and Kokoro-82M architectures.
Refines coarse acoustic predictions by capturing temporal and spectral correlations.
"""
import torch
import torch.nn as nn


class PostNet(nn.Module):
    """
    5-layer Conv1D PostNet for Mel-Spectrogram harmonic and formant refinement.
    Input:  [B, T_mel, 80] (coarse mel)
    Output: [B, T_mel, 80] (residual)
    """
    def __init__(
        self,
        mel_dim: int = 80,
        postnet_dim: int = 256,
        n_layers: int = 5,
        kernel_size: int = 5,
        dropout: float = 0.2
    ):
        super().__init__()
        self.convolutions = nn.ModuleList()

        # First layer: mel_dim -> postnet_dim
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    mel_dim, postnet_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    bias=True
                ),
                nn.GroupNorm(1, postnet_dim),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        )

        # Intermediate layers: postnet_dim -> postnet_dim
        for _ in range(n_layers - 2):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(
                        postnet_dim, postnet_dim,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=True
                    ),
                    nn.GroupNorm(1, postnet_dim),
                    nn.Tanh(),
                    nn.Dropout(dropout)
                )
            )

        # Final projection layer: postnet_dim -> mel_dim (no activation for residual)
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(
                    postnet_dim, mel_dim,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    bias=True
                ),
                nn.Dropout(dropout)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T_mel, 80]
        Returns: [B, T_mel, 80]
        """
        # Transpose to [B, 80, T_mel] for Conv1D
        x_conv = x.transpose(1, 2)
        for conv in self.convolutions:
            x_conv = conv(x_conv)
        # Transpose back to [B, T_mel, 80]
        return x_conv.transpose(1, 2)
