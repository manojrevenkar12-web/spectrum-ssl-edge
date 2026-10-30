"""Complex-baseband waveform generators.

Every generator returns `n` windows of `length` samples at `sps` samples per symbol, normalised to unit
average power. Known classes are used for training; novel classes appear only at test time (open set).
All generators are vectorised over the batch and draw randomness only from the `rng` they are given.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.signal import lfilter

KNOWN = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "GFSK", "AM-DSB", "WBFM", "OFDM"]
NOVEL = ["OQPSK", "32QAM", "CSS", "CW"]

RRC_BETA = 0.35
RRC_SPAN = 8  # symbols


def rrc_taps(sps: int, beta: float = RRC_BETA, span: int = RRC_SPAN) -> np.ndarray:
    """Root-raised-cosine filter, unit energy."""
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    h = np.empty_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1 / (4 * beta)) < 1e-12:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (
                np.pi * ti * (1 - (4 * beta * ti) ** 2)
            )
    return h / np.sqrt(np.sum(h**2))


def _constellation(name: str) -> np.ndarray:
    if name == "BPSK":
        c = np.array([1, -1], dtype=complex)
    elif name in ("QPSK", "OQPSK"):
        c = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    elif name == "8PSK":
        c = np.exp(1j * 2 * np.pi * np.arange(8) / 8)
    elif name == "PAM4":
        c = np.array([-3, -1, 1, 3], dtype=complex)
    elif name == "16QAM":
        g = np.arange(-3, 4, 2)
        c = (g[:, None] + 1j * g[None, :]).ravel()
    elif name == "64QAM":
        g = np.arange(-7, 8, 2)
        c = (g[:, None] + 1j * g[None, :]).ravel()
    elif name == "32QAM":  # cross constellation: 6x6 grid without the four corners
        g = np.arange(-5, 6, 2)
        grid = (g[:, None] + 1j * g[None, :]).ravel()
        c = grid[~((np.abs(grid.real) == 5) & (np.abs(grid.imag) == 5))]
    else:
        raise ValueError(name)
    return c / np.sqrt(np.mean(np.abs(c) ** 2))


def _normalise(x: np.ndarray) -> np.ndarray:
    p = np.mean(np.abs(x) ** 2, axis=1, keepdims=True)
    return x / np.sqrt(np.maximum(p, 1e-12))


def _window(x: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Random start per row so symbol timing is uniformly distributed."""
    n, total = x.shape
    start = rng.integers(0, total - length + 1, size=n)
    idx = start[:, None] + np.arange(length)[None, :]
    return np.take_along_axis(x, idx, axis=1)


def _linear(name: str, n: int, length: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    span = RRC_SPAN
    nsym = length // sps + 2 * span + 2
    c = _constellation(name)
    sym = c[rng.integers(0, len(c), size=(n, nsym))]
    up = np.zeros((n, nsym * sps), dtype=complex)
    up[:, ::sps] = sym
    if name == "OQPSK":  # delay the quadrature rail by half a symbol
        q = np.zeros_like(up.imag)
        q[:, sps // 2 :] = up.imag[:, : -(sps // 2)]
        up = up.real + 1j * q
    h = rrc_taps(sps)
    y = lfilter(h, 1.0, up, axis=1)
    y = y[:, span * sps : -span * sps]
    return _normalise(_window(y, length, rng))


def _gaussian_taps(sps: int, bt: float = 0.35, span: int = 4) -> np.ndarray:
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    alpha = np.sqrt(np.log(2) / 2) / bt
    h = np.sqrt(np.pi) / alpha * np.exp(-((np.pi * t / alpha) ** 2))
    return h / np.sum(h)


def _gfsk(n: int, length: int, sps: int, rng: np.random.Generator, h_index: float = 0.5) -> np.ndarray:
    nsym = length // sps + 8
    bits = rng.choice([-1.0, 1.0], size=(n, nsym))
    f = np.repeat(bits, sps, axis=1)
    f = lfilter(_gaussian_taps(sps), 1.0, f, axis=1)
    phase = np.pi * h_index * np.cumsum(f, axis=1) / sps
    return _normalise(_window(np.exp(1j * phase), length, rng))


def _message(n: int, total: int, rng: np.random.Generator, cutoff: float = 0.02) -> np.ndarray:
    """Band-limited random message (stands in for audio), unit variance."""
    w = rng.standard_normal((n, total + 64))
    a = np.exp(-2 * np.pi * cutoff)
    m = lfilter([1 - a], [1, -a], w, axis=1)
    m = lfilter([1 - a], [1, -a], m, axis=1)[:, 64:]
    return m / np.maximum(m.std(axis=1, keepdims=True), 1e-9)


def _am_dsb(n: int, length: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    m = _message(n, length, rng)
    depth = rng.uniform(0.3, 0.8, size=(n, 1))
    return _normalise((1 + depth * np.clip(m, -1.5, 1.5) / 1.5).astype(complex))


def _wbfm(n: int, length: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    m = _message(n, length, rng)
    dev = rng.uniform(0.04, 0.08, size=(n, 1))  # peak-ish deviation, cycles/sample
    phase = 2 * np.pi * np.cumsum(dev * m, axis=1)
    return _normalise(np.exp(1j * phase))


def _ofdm(n: int, length: int, sps: int, rng: np.random.Generator, nfft: int = 32, cp: int = 8) -> np.ndarray:
    used = np.r_[np.arange(-12, 0), np.arange(1, 13)]  # 24 of 32 subcarriers, DC empty
    nsym = length // (nfft + cp) + 2
    qpsk = _constellation("QPSK")
    grid = np.zeros((n, nsym, nfft), dtype=complex)
    grid[:, :, used % nfft] = qpsk[rng.integers(0, 4, size=(n, nsym, len(used)))]
    t = np.fft.ifft(grid, axis=2)
    t = np.concatenate([t[:, :, -cp:], t], axis=2).reshape(n, -1)
    return _normalise(_window(t, length, rng))


def _css(n: int, length: int, sps: int, rng: np.random.Generator, sf: int = 6) -> np.ndarray:
    """Chirp spread spectrum (LoRa-like): each symbol is a cyclically shifted up-chirp."""
    chips = 2**sf
    nsym = length // chips + 2
    k = np.arange(chips)
    sym = rng.integers(0, chips, size=(n, nsym))
    shifted = (k[None, None, :] + sym[:, :, None]) % chips
    phase = 2 * np.pi * (shifted**2 / (2 * chips) - shifted / 2)
    return _normalise(_window(np.exp(1j * phase).reshape(n, -1), length, rng))


def _cw(n: int, length: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    f = rng.uniform(-0.05, 0.05, size=(n, 1))
    return np.exp(1j * (2 * np.pi * f * np.arange(length)[None, :] + rng.uniform(0, 2 * np.pi, size=(n, 1))))


def _linear_gen(name: str) -> Callable[..., np.ndarray]:
    def gen(n: int, length: int, sps: int, rng: np.random.Generator) -> np.ndarray:
        return _linear(name, n, length, sps, rng)

    return gen


GENERATORS: dict[str, Callable[..., np.ndarray]] = {
    **{c: _linear_gen(c) for c in ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "OQPSK", "32QAM"]},
    "GFSK": _gfsk,
    "AM-DSB": _am_dsb,
    "WBFM": _wbfm,
    "OFDM": _ofdm,
    "CSS": _css,
    "CW": _cw,
}


def generate(name: str, n: int, length: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    if n == 0:
        return np.zeros((0, length), dtype=np.complex128)
    return GENERATORS[name](n, length, sps, rng)
