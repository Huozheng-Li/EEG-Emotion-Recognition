"""
EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs.
Reference: Lawhern et al., J. Neural Eng. 2018

~2,500 parameters — designed for small-sample EEG classification.
"""
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    EEGNet for EEG emotion classification.

    Args:
        n_channels: number of EEG channels (default 30)
        n_times: number of time points per trial
        n_classes: number of output classes
        F1: number of temporal filters (default 8)
        D: depth multiplier for spatial filters (default 2)
        F2: number of pointwise filters (default 16)
        dropout: dropout rate
    """

    def __init__(self, n_channels: int = 30, n_times: int = 1250,
                 n_classes: int = 2, F1: int = 8, D: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        F2 = F1 * D

        # Block 1: Temporal → Spatial
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, 64), padding="same", bias=False),
            nn.BatchNorm2d(F1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1), groups=F1,
                      bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: Separable Conv
        self.sep_conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(F2, F2, kernel_size=(1, 16), groups=F2,
                      padding="same", bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._feat_dim(n_channels, n_times, F2), n_classes),
        )

    def _feat_dim(self, C, T, F2):
        # Block 1: temporal (same padding) → AvgPool(1,4) → T//4
        T1 = T // 4
        # Block 2: sep_conv (same padding) → AvgPool(1,8) → T1//8
        T2 = T1 // 8
        return F2 * T2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.sep_conv(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = EEGNet(n_channels=30, n_times=1250, n_classes=2)
    print(f"EEGNet params: {count_parameters(model):,}")

    x = torch.randn(2, 30, 1250)
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
