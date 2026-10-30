"""Labelled windows for a domain: (N, 2, L) float32 I/Q, (N, L) mask, labels, per-window parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import Domain, impair
from .modulate import KNOWN, generate


@dataclass
class Windows:
    x: np.ndarray  # (N, 2, L) float32
    mask: np.ndarray  # (N, L) bool
    y: np.ndarray  # (N,) int64; index into KNOWN, -1 for a novel class
    name: np.ndarray  # (N,) class names
    meta: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> Windows:
        return Windows(
            self.x[idx], self.mask[idx], self.y[idx], self.name[idx], {k: v[idx] for k, v in self.meta.items()}
        )

    @staticmethod
    def concat(parts: list[Windows]) -> Windows:
        keys = set.intersection(*(set(p.meta) for p in parts)) if parts else set()
        return Windows(
            np.concatenate([p.x for p in parts]),
            np.concatenate([p.mask for p in parts]),
            np.concatenate([p.y for p in parts]),
            np.concatenate([p.name for p in parts]),
            {k: np.concatenate([p.meta[k] for p in parts]) for k in keys},
        )


def make_windows(classes: list[str], per_class: int, dom: Domain, length: int, seed: int) -> Windows:
    """Deterministic in (classes, per_class, domain, length, seed)."""
    rng = np.random.default_rng(seed)
    parts: list[Windows] = []
    for c in classes:
        sps = rng.choice(np.asarray(dom.sps), size=per_class)
        clean = np.empty((per_class, length), dtype=np.complex128)
        for s in np.unique(sps):
            idx = np.nonzero(sps == s)[0]
            clean[idx] = generate(c, len(idx), length, int(s), rng)
        rx, mask, meta = impair(clean, dom, rng)
        meta["sps"] = sps.astype(float)
        x = np.stack([rx.real, rx.imag], axis=1).astype(np.float32)
        label = KNOWN.index(c) if c in KNOWN else -1
        parts.append(Windows(x, mask, np.full(per_class, label, dtype=np.int64), np.full(per_class, c), meta))
    w = Windows.concat(parts)
    order = rng.permutation(len(w))
    return w.subset(order)
