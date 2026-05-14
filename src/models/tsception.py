"""
TSception: Capturing Temporal Dynamics and Spatial Asymmetry
from EEG for Emotion Recognition.

Reference: Ding et al., IEEE TAC 2023
https://arxiv.org/abs/2104.02935
"""
import torch
import torch.nn as nn
import numpy as np


class DynamicTemporalLayer(nn.Module):
    """
    Multi-scale temporal convolution.

    Uses num_T parallel 1D conv branches with different kernel sizes
    to capture EEG dynamics at different time scales.
    """

    def __init__(self, n_channels: int, num_T: int, hid_channels: int,
                 sampling_rate: int):
        super().__init__()
        self.n_channels = n_channels
        self.num_T = num_T
        self.hid_channels = hid_channels

        # Kernel sizes: spans different frequency ranges
        # e.g., for 250 Hz: [125, 64, 32, 16, 8, 4, ...]
        kernels = []
        for i in range(num_T):
            k = int(sampling_rate / (2.0 + i * 2.0))
            if k < 2:
                k = 2
            kernels.append(k)

        # Deduplicate and sort descending
        kernels = sorted(set(kernels), reverse=True)
        self.num_T = len(kernels)
        self.kernels = kernels

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, hid_channels, kernel_size=(1, k), padding="same",
                          bias=False),
                nn.BatchNorm2d(hid_channels),
            )
            for k in kernels
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_channels, n_times)
        Returns:
            (B, num_T * hid_channels, n_channels, n_times)
        """
        outs = []
        for conv in self.convs:
            out = conv(x)  # (B, hid_channels, n_channels, n_times)
            outs.append(out)
        return torch.cat(outs, dim=1)


class AsymmetricSpatialLayer(nn.Module):
    """
    Asymmetric spatial convolution capturing global + hemisphere-specific
    EEG patterns using 1D convolutions along the channel axis.
    """

    def __init__(self, in_channels: int, n_channels: int, num_S: int):
        """
        Args:
            in_channels: num_T * hid_channels from temporal layer
            n_channels: number of EEG channels (e.g., 30)
            num_S: number of spatial branches
        """
        super().__init__()
        self.n_channels = n_channels
        self.num_S = num_S
        self.out_ch = in_channels // 4

        self.global_conv = nn.Conv2d(in_channels, self.out_ch,
                                     kernel_size=(n_channels, 1),
                                     padding=0, bias=False)
        self.bn_global = nn.BatchNorm2d(self.out_ch)

        left_ch = n_channels // 2
        right_ch = n_channels - left_ch
        self.left_conv = nn.Conv2d(in_channels, self.out_ch,
                                   kernel_size=(left_ch, 1),
                                   padding=0, bias=False)
        self.bn_left = nn.BatchNorm2d(self.out_ch)
        self.right_conv = nn.Conv2d(in_channels, self.out_ch,
                                    kernel_size=(right_ch, 1),
                                    padding=0, bias=False)
        self.bn_right = nn.BatchNorm2d(self.out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_T*hid, n_channels, n_times)
        Returns:
            (B, 3*out_ch, 1, n_times)
        """
        B, _, C, T = x.shape
        left_ch = C // 2

        g = self.bn_global(self.global_conv(x))       # (B, out_ch, 1, T)

        x_left = x[:, :, :left_ch, :]                  # (B, in_ch, left_ch, T)
        l = self.bn_left(self.left_conv(x_left))       # (B, out_ch, 1, T)

        x_right = x[:, :, left_ch:, :]                 # (B, in_ch, right_ch, T)
        r = self.bn_right(self.right_conv(x_right))    # (B, out_ch, 1, T)

        return torch.cat([g, l, r], dim=1)  # (B, 3*out_ch, 1, T)


class TSception(nn.Module):
    """
    TSception for EEG emotion recognition.

    Args:
        n_channels: number of EEG channels
        n_times: number of time points per trial
        num_T: temporal branches
        num_S: spatial branches
        hid_channels: hidden channels per temporal branch
        num_classes: output classes
        dropout: dropout rate
        sampling_rate: EEG sampling rate in Hz
    """

    def __init__(self, n_channels: int = 30, n_times: int = 2500,
                 num_T: int = 15, num_S: int = 15,
                 hid_channels: int = 32, num_classes: int = 2,
                 dropout: float = 0.5, sampling_rate: int = 250):
        super().__init__()

        self.temporal = DynamicTemporalLayer(n_channels, num_T,
                                             hid_channels, sampling_rate)
        # After temporal: (B, actual_T * hid_channels, n_channels, n_times)

        T_actual = self.temporal.num_T
        spatial_in = T_actual * hid_channels

        self.spatial = AsymmetricSpatialLayer(spatial_in, n_channels, num_S)
        # After spatial: (B, 3 * spatial_in/4, 1, n_times)

        self.dropout = nn.Dropout(dropout)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 4))  # compress time to 4

        spatial_out = 3 * (spatial_in // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(spatial_out * 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_channels, n_times) or (B, 1, n_channels, n_times)
        Returns:
            logits: (B, num_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, n_channels, n_times)

        x = self.temporal(x)       # (B, T*hid, C, T)
        x = self.spatial(x)        # (B, 3*spatial_out, 1, T)
        x = self.dropout(x)
        x = self.avg_pool(x)       # (B, 3*spatial_out, 1, 4)
        x = self.classifier(x)     # (B, num_classes)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test forward pass
    model = TSception(n_channels=30, n_times=2500, sampling_rate=250,
                      num_classes=2)
    print(f"TSception params: {count_parameters(model):,}")

    x = torch.randn(2, 30, 2500)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
