"""Batched, RF-aware augmentations on (N, 2, L) I/Q tensors with (N, L) masks.

Each one is a nuisance the label should not depend on: carrier phase, a small frequency offset, where
the window starts, how much of the burst was captured, extra thermal noise, lost samples.
"""

from __future__ import annotations

from typing import Any

import torch


def _complex(x: torch.Tensor) -> torch.Tensor:
    return torch.complex(x[:, 0], x[:, 1])


def _real(z: torch.Tensor) -> torch.Tensor:
    return torch.stack([z.real, z.imag], dim=1)


def renormalise(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    p = (x.pow(2).sum(dim=1) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return x / p.clamp_min(1e-12).sqrt()[:, None, None]


def augment(
    x: torch.Tensor, mask: torch.Tensor, spec: dict[str, Any], g: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    n, _, length = x.shape
    dev = x.device
    z = _complex(x)
    m = mask.clone()
    t = torch.arange(length, device=dev, dtype=torch.float32)[None, :]

    if spec.get("phase"):
        theta = torch.rand(n, 1, generator=g, device=dev) * 2 * torch.pi
        z = z * torch.polar(torch.ones_like(theta), theta)
    if spec.get("cfo"):
        f = (torch.rand(n, 1, generator=g, device=dev) * 2 - 1) * float(spec["cfo"])
        z = z * torch.polar(torch.ones_like(t * f), 2 * torch.pi * f * t)
    if spec.get("shift"):
        s = int(spec["shift"])
        r = torch.randint(-s, s + 1, (n,), generator=g, device=dev)
        idx = (torch.arange(length, device=dev)[None, :] - r[:, None]) % length
        z = torch.gather(z, 1, idx)
        m = torch.gather(m, 1, idx)
    if spec.get("crop"):
        lo, hi = spec["crop"]
        c = torch.randint(int(lo), int(hi) + 1, (n,), generator=g, device=dev)
        start = (torch.rand(n, generator=g, device=dev) * (length - c + 1).float()).long()
        idx = (start[:, None] + torch.arange(length, device=dev)[None, :]).clamp_max(length - 1)
        keep = torch.arange(length, device=dev)[None, :] < c[:, None]
        z = torch.gather(z, 1, idx) * keep
        m = torch.gather(m, 1, idx) * keep
    if spec.get("noise_snr_db"):
        lo, hi = spec["noise_snr_db"]
        snr = lo + torch.rand(n, 1, generator=g, device=dev) * (hi - lo)
        sigma = torch.sqrt(10 ** (-snr / 10) / 2)
        noise = torch.complex(
            torch.randn(n, length, generator=g, device=dev), torch.randn(n, length, generator=g, device=dev)
        )
        z = z + noise * sigma * m
    if spec.get("dropout"):
        lo, hi = spec["dropout"]
        frac = lo + torch.rand(n, generator=g, device=dev) * (hi - lo)
        w = (frac * length).long()
        st = (torch.rand(n, generator=g, device=dev) * (length - w + 1).float()).long()
        ar = torch.arange(length, device=dev)[None, :]
        gap = (ar >= st[:, None]) & (ar < (st + w)[:, None])
        m = m * (~gap)
        z = z * (~gap)
    out = _real(z) * m[:, None, :]
    return renormalise(out, m), m
