import torch
import torch.nn as nn

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class PureCNN(nn.Module):
    """
    1D 纯 CNN：4 个卷积块 + GAP + 全连接
    通过 width_mult / fc_dim 调整参数量，便于对齐到主模型 ±10%
    """
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels=(32, 64, 128, 128),
        kernel_sizes=(7, 5, 5, 3),
        strides=(4, 2, 2, 2),
        fc_dim: int = 128,
        dropout: float = 0.5,
        width_mult: float = 1.0,
    ):
        super().__init__()
        chs = [max(1, int(c * width_mult)) for c in conv_channels]

        layers = []
        prev = in_channels
        for c, k, s in zip(chs, kernel_sizes, strides):
            layers += [
                nn.Conv1d(prev, c, kernel_size=k, stride=s, padding=k // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
            ]
            prev = c
        self.feature = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1),
        )

    def forward(self, x):  # x: (B, 1, T)
        x = self.feature(x)
        x = self.gap(x)               # (B, C, 1)
        logits = self.classifier(x)   # (B, 1)
        return logits.squeeze(-1) 