"""Transmitter, channel and receiver impairments, and the irregular-sampling effects.

A *domain* is a set of parameter ranges (config/domains.yaml). Each window draws its own parameters
uniformly from those ranges, so a domain is a distribution, not a fixed setting. Order follows the
signal path: transmitter power amplifier -> multipath channel -> receiver LO (CFO, phase noise) ->
thermal noise -> IQ imbalance and DC offset -> what the recorder kept (bursts, dropouts, length).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _rng_range(rng: np.random.Generator, r: list[float] | tuple[float, float], n: int) -> np.ndarray:
    lo, hi = float(r[0]), float(r[-1])
    return rng.uniform(lo, hi, size=n) if hi > lo else np.full(n, lo)


@dataclass(frozen=True)
class Domain:
    name: str
    snr_db: tuple[float, float] = (-4.0, 20.0)
    sps: tuple[int, ...] = (8,)
    cfo: tuple[float, float] = (-0.002, 0.002)
    phase_noise: tuple[float, float] = (0.0, 0.002)
    iq_amp_db: tuple[float, float] = (0.0, 0.5)
    iq_phase_deg: tuple[float, float] = (0.0, 2.0)
    dc: tuple[float, float] = (0.0, 0.02)
    pa_backoff_db: tuple[float, float] | None = None
    pa_smoothness: float = 2.0
    multipath_taps: tuple[int, int] | None = None
    multipath_decay: tuple[float, float] = (1.0, 3.0)
    multipath_k_db: tuple[float, float] = (0.0, 6.0)
    burst_occupancy: tuple[float, float] | None = None
    dropout: tuple[float, float] = (0.0, 0.0)
    valid: tuple[int, int] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(name: str, d: dict[str, Any]) -> Domain:
        known = set(Domain.__dataclass_fields__) - {"name", "extra"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"domain {name}: unknown parameters {sorted(unknown)}")
        kw: dict[str, Any] = {}
        for k, v in d.items():
            if v is None:
                kw[k] = None
            elif isinstance(v, list):
                kw[k] = tuple(v)
            else:
                kw[k] = v
        dom = Domain(name=name, **kw)
        dom.validate()
        return dom

    def validate(self) -> None:
        def ordered(r: tuple | None, what: str, lo: float, hi: float) -> None:
            if r is None:
                return
            if len(r) not in (1, 2) or r[0] > r[-1]:
                raise ValueError(f"{self.name}.{what}: expected [low, high], got {r}")
            if r[0] < lo or r[-1] > hi:
                raise ValueError(f"{self.name}.{what}: {r} outside the supported range [{lo}, {hi}]")

        ordered(self.snr_db, "snr_db", -20, 40)
        ordered(self.cfo, "cfo", -0.05, 0.05)
        ordered(self.phase_noise, "phase_noise", 0, 0.05)
        ordered(self.iq_amp_db, "iq_amp_db", 0, 3)
        ordered(self.iq_phase_deg, "iq_phase_deg", 0, 15)
        ordered(self.dc, "dc", 0, 0.3)
        ordered(self.pa_backoff_db, "pa_backoff_db", 0, 12)
        ordered(self.multipath_taps, "multipath_taps", 1, 8)
        ordered(self.multipath_decay, "multipath_decay", 0.2, 8)
        ordered(self.multipath_k_db, "multipath_k_db", -10, 20)
        ordered(self.burst_occupancy, "burst_occupancy", 0.1, 1)
        ordered(self.dropout, "dropout", 0, 0.3)
        ordered(self.valid, "valid", 32, 4096)
        if not self.sps or any(s < 2 or s > 32 for s in self.sps):
            raise ValueError(f"{self.name}.sps: values must be in [2, 32]")


def power_amplifier(x: np.ndarray, backoff_db: np.ndarray, p: float) -> np.ndarray:
    """Rapp model: AM/AM compression, no AM/PM. Saturation relative to the window's RMS (=1)."""
    a = 10 ** (backoff_db[:, None] / 20)
    r = np.abs(x)
    g = 1.0 / (1 + (r / a) ** (2 * p)) ** (1 / (2 * p))
    y = x * g
    return y / np.sqrt(np.mean(np.abs(y) ** 2, axis=1, keepdims=True))


def multipath(
    x: np.ndarray, taps: np.ndarray, decay: np.ndarray, k_db: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Tapped delay line, exponential power-delay profile, Rician first tap. Total power 1."""
    n, length = x.shape
    tmax = int(taps.max())
    k = np.arange(tmax)[None, :]
    pdp = np.exp(-k / decay[:, None]) * (k < taps[:, None])
    pdp /= pdp.sum(axis=1, keepdims=True)
    z = (rng.standard_normal((n, tmax)) + 1j * rng.standard_normal((n, tmax))) / np.sqrt(2)
    h = z * np.sqrt(pdp)
    kl = 10 ** (k_db / 10)
    los = np.exp(1j * rng.uniform(0, 2 * np.pi, n))
    h[:, 0] = np.sqrt(pdp[:, 0]) * (np.sqrt(kl / (kl + 1)) * los + np.sqrt(1 / (kl + 1)) * z[:, 0])
    h /= np.sqrt(np.sum(np.abs(h) ** 2, axis=1, keepdims=True))
    y = np.zeros_like(x)
    for d in range(tmax):
        y[:, d:] += h[:, d : d + 1] * x[:, : length - d]
    return y


def receiver_lo(x: np.ndarray, cfo: np.ndarray, pn_std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n, length = x.shape
    t = np.arange(length)[None, :]
    walk = np.cumsum(rng.standard_normal((n, length)) * pn_std[:, None], axis=1)
    phase = 2 * np.pi * cfo[:, None] * t + rng.uniform(0, 2 * np.pi, (n, 1)) + walk
    return x * np.exp(1j * phase)


def iq_frontend(
    x: np.ndarray, amp_db: np.ndarray, phase_deg: np.ndarray, dc: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    g = 10 ** (amp_db[:, None] / 20)
    phi = np.deg2rad(phase_deg)[:, None]
    i, q = x.real, x.imag
    q2 = g * (np.cos(phi) * q + np.sin(phi) * i)
    dc_c = dc[:, None] * np.exp(1j * rng.uniform(0, 2 * np.pi, (len(dc), 1)))
    return i + 1j * q2 + dc_c


def impair(
    clean: np.ndarray, dom: Domain, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Returns (received windows, validity mask, per-window parameters)."""
    n, length = clean.shape
    meta: dict[str, np.ndarray] = {}
    x = clean
    if dom.pa_backoff_db is not None:
        b = _rng_range(rng, dom.pa_backoff_db, n)
        x = power_amplifier(x, b, dom.pa_smoothness)
        meta["pa_backoff_db"] = b
    if dom.multipath_taps is not None:
        taps = rng.integers(int(dom.multipath_taps[0]), int(dom.multipath_taps[-1]) + 1, size=n)
        decay = _rng_range(rng, dom.multipath_decay, n)
        kdb = _rng_range(rng, dom.multipath_k_db, n)
        x = multipath(x, taps, decay, kdb, rng)
        meta["multipath_taps"] = taps.astype(float)

    # bursts: the emitter is on for part of the window only; noise is everywhere
    on = np.ones((n, length), dtype=bool)
    if dom.burst_occupancy is not None:
        occ = _rng_range(rng, dom.burst_occupancy, n)
        w = np.maximum((occ * length).astype(int), 16)
        s = (rng.uniform(0, 1, n) * (length - w + 1)).astype(int)
        t = np.arange(length)[None, :]
        on = (t >= s[:, None]) & (t < (s + w)[:, None])
        x = x * on
        meta["occupancy"] = occ

    cfo = _rng_range(rng, dom.cfo, n)
    pn = _rng_range(rng, dom.phase_noise, n)
    x = receiver_lo(x, cfo, pn, rng)
    snr = _rng_range(rng, dom.snr_db, n)
    noise_p = 10 ** (-snr / 10)
    noise = (rng.standard_normal((n, length)) + 1j * rng.standard_normal((n, length))) * np.sqrt(noise_p[:, None] / 2)
    x = x + noise
    x = iq_frontend(
        x, _rng_range(rng, dom.iq_amp_db, n), _rng_range(rng, dom.iq_phase_deg, n), _rng_range(rng, dom.dc, n), rng
    )

    mask = np.ones((n, length), dtype=bool)
    if dom.dropout[-1] > 0:
        frac = _rng_range(rng, dom.dropout, n)
        for i in np.nonzero(frac > 0)[0]:
            gaps = rng.integers(1, 5)
            total = int(round(frac[i] * length))
            sizes = np.maximum(1, rng.multinomial(total, np.ones(gaps) / gaps)) if total else []
            for size in sizes:
                st = rng.integers(0, max(1, length - size))
                mask[i, st : st + size] = False
        meta["dropout"] = frac
    if dom.valid is not None:
        v = rng.integers(int(dom.valid[0]), int(min(dom.valid[-1], length)) + 1, size=n)
        mask &= np.arange(length)[None, :] < v[:, None]
        meta["valid"] = v.astype(float)
    x = np.where(mask, x, 0)

    # the network sees a gain-normalised input, as an AGC would deliver it
    p = np.sum(np.abs(x) ** 2, axis=1) / np.maximum(mask.sum(axis=1), 1)
    x = x / np.sqrt(np.maximum(p, 1e-12))[:, None]
    meta.update(snr_db=snr, cfo=cfo)
    return x, mask, meta
