"""A small 1-D ResNet over I/Q with masked pooling, sized for an embedded int8 budget (~2.5 M MAC/window).

The architecture is deliberately limited to what the int8 runtime (`edge-rs`) implements: Conv1d with
"same" padding and a stride, BatchNorm (folded into the conv for export), ReLU, residual adds, masked
mean and max pooling over time, and Linear layers. Anything added here must be added there too, and
`tests/test_edge_parity.py` will fail until it is.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

STEM = (24, 7, 2)  # channels, kernel, stride
BLOCKS = [(24, 1), (32, 2), (48, 2), (64, 2)]  # out channels, stride
KERNEL = 5
EMBED = 128


def downsample_mask(mask: torch.Tensor, stride: int) -> torch.Tensor:
    """A step is valid if any input sample in its stride block was valid. mask: (N, L) float."""
    if stride == 1:
        return mask
    n, length = mask.shape
    pad = (-length) % stride
    if pad:
        mask = F.pad(mask, (0, pad))
    return mask.view(n, -1, stride).amax(dim=2)


class ConvBN(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int, stride: int) -> None:
        super().__init__(nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False), nn.BatchNorm1d(cout))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.c1 = ConvBN(cin, cout, KERNEL, stride)
        self.c2 = ConvBN(cout, cout, KERNEL, 1)
        self.short = ConvBN(cin, cout, 1, stride) if (stride != 1 or cin != cout) else None
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.c1(x))
        h = self.c2(h)
        s = x if self.short is None else self.short(x)
        return F.relu(h + s)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c, k, s = STEM
        self.stem = ConvBN(2, c, k, s)
        blocks = []
        cin = c
        for cout, stride in BLOCKS:
            blocks.append(ResBlock(cin, cout, stride))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Linear(2 * cin, EMBED)
        self.total_stride = s
        for _, stride in BLOCKS:
            self.total_stride *= stride

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.stem(x))
        for b in self.blocks:
            h = b(h)
        return h

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = downsample_mask(mask, self.total_stride)[:, None, : h.shape[2]]
        count = m.sum(dim=2).clamp_min(1.0)
        mean = (h * m).sum(dim=2) / count
        mx = h.masked_fill(m == 0, float("-inf")).amax(dim=2)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        return torch.cat([mean, mx], dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.proj(self.pool(self.features(x), mask))


class Classifier(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(EMBED, n_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x, mask))

    def embed_and_logits(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, mask)
        return z, self.head(z)


class Projector(nn.Module):
    """SimCLR projection head; discarded after pretraining."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(EMBED, EMBED), nn.ReLU(), nn.Linear(EMBED, dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def count_macs(length: int) -> int:
    """Multiply-accumulates per window, for the edge budget."""
    c, k, s = STEM
    L = (length + s - 1) // s
    macs = 2 * c * k * L
    cin = c
    for cout, stride in BLOCKS:
        L = (L + stride - 1) // stride
        macs += cin * cout * KERNEL * L + cout * cout * KERNEL * L
        if stride != 1 or cin != cout:
            macs += cin * cout * L
        cin = cout
    return macs + 2 * cin * EMBED
