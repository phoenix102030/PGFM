"""Spatial residual and resolution-changing blocks used by the flow model."""

import torch

import torch.nn as nn

import torch.nn.functional as F

def _match_size_2d(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    _, _, H, W = x.shape
    _, _, Hr, Wr = ref.shape
    if H > Hr:
        dh = (H - Hr) // 2
        x = x[:, :, dh:dh + Hr, :]
    if W > Wr:
        dw = (W - Wr) // 2
        x = x[:, :, :, dw:dw + Wr]
    pad_h = Hr - x.shape[2]
    pad_w = Wr - x.shape[3]
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, max(0, pad_w), 0, max(0, pad_h)))
    return x

class ResBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8, dropout: float = 0.0):
        super().__init__()
        g1 = min(groups, in_ch)
        g2 = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(num_groups=g1, num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=g2, num_channels=out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

class Down2D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Up2D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
