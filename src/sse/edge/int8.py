"""Integer-only inference: the reference the Rust runtime must match bit for bit.

Scheme (per-channel symmetric int8 weights, per-tensor symmetric int8 activations, int32 accumulators):
  * BatchNorm is folded into each convolution.
  * Every requantisation is `(acc * m + 2^(s-1)) >> s` in 64-bit integers, with (m, s) chosen so that
    m / 2^s approximates the float multiplier and m lies in [2^30, 2^31). No floating point after the
    input is quantised, so two implementations either agree exactly or one of them is wrong.
  * A residual block requantises both branches to the block's output scale as int32, adds, and applies
    ReLU as a clamp to [0, 127]; nothing saturates before the add.
  * Masked mean pooling divides the int32 sum by the number of valid steps with round-half-up; masked
    max takes the int8 maximum. Both stay at the last block's scale.
  * The classifier head returns int32 accumulators; logits = acc * scale[c] (only for reporting).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np
import torch

from ..models.encoder import Classifier, ConvBN

QMAX = 127


def fixed_point(multiplier: float) -> tuple[int, int]:
    """(m, s) with m / 2^s ~= multiplier, m in [2^30, 2^31)."""
    if multiplier <= 0:
        return 0, 1
    m, e = np.frexp(multiplier)  # multiplier = m * 2^e, m in [0.5, 1)
    q = int(round(m * (1 << 31)))
    s = 31 - int(e)
    if q == 1 << 31:
        q //= 2
        s -= 1
    if s < 1 or s > 62:
        raise ValueError(f"multiplier {multiplier} out of range for 64-bit requantisation")
    return q, s


def requant(acc: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Round-half-up fixed-point rescale of int accumulators; m and s broadcast over channels."""
    acc = acc.astype(np.int64)
    m = np.asarray(m, dtype=np.int64)
    s = np.asarray(s, dtype=np.int64)
    return (acc * m + (np.int64(1) << (s - 1))) >> s


def quantise_input(x: np.ndarray, inv_scale: np.float32) -> np.ndarray:
    """float32 -> int8 with floor(x * inv_scale + 0.5), all in float32 (the Rust code does the same)."""
    v = np.floor(x.astype(np.float32) * np.float32(inv_scale) + np.float32(0.5))
    return np.clip(v, -QMAX, QMAX).astype(np.int64)


@dataclass
class QConv:
    w: np.ndarray  # (cout, cin, k) int8
    b: np.ndarray  # (cout,) int32
    stride: int
    m: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    s: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))


@dataclass
class QBlock:
    c1: QConv
    c2: QConv
    short: QConv | None
    id_m: int = 0  # identity shortcut rescale
    id_s: int = 1


@dataclass
class QLinear:
    w: np.ndarray  # (out, in) int8
    b: np.ndarray  # (out,) int32
    m: np.ndarray | None = None
    s: np.ndarray | None = None
    out_scale: np.ndarray | None = None  # head only: float scale per output


@dataclass
class QModel:
    length: int
    in_inv_scale: np.float32
    stem: QConv
    blocks: list[QBlock]
    total_stride: int
    proj: QLinear
    head: QLinear
    scales: dict[str, float]

    @property
    def weight_bytes(self) -> int:
        n = self.stem.w.size + 4 * self.stem.b.size
        for b in self.blocks:
            for c in (b.c1, b.c2, b.short):
                if c is not None:
                    n += c.w.size + 4 * c.b.size
        return n + self.proj.w.size + 4 * self.proj.b.size + self.head.w.size + 4 * self.head.b.size


# --- float side: folding and calibration ----------------------------------------------------------


def fold(cb: ConvBN) -> tuple[np.ndarray, np.ndarray, int]:
    conv, bn = cb[0], cb[1]
    w = conv.weight.detach().double().numpy()
    g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double().numpy()
    b = (bn.bias - bn.running_mean * bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double().numpy()
    return w * g[:, None, None], b, conv.stride[0]


def _conv_f(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int) -> np.ndarray:
    """Float conv1d, 'same' padding k//2, on (N, C, L) float64; mirrors the int version."""
    return _conv_core(x, w, stride) + b[None, :, None]


def _conv_core(x: np.ndarray, w: np.ndarray, stride: int) -> np.ndarray:
    n, cin, length = x.shape
    cout, _, k = w.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (0, 0), (p, p)))
    lout = (length + 2 * p - k) // stride + 1
    idx = np.arange(lout)[:, None] * stride + np.arange(k)[None, :]
    cols = xp[:, :, idx]  # (n, cin, lout, k)
    return np.einsum("nclk,ock->nol", cols, w, optimize=True)


def _mask_ds(mask: np.ndarray, stride: int, steps: int) -> np.ndarray:
    n, length = mask.shape
    pad = (-length) % stride
    m = np.pad(mask, ((0, 0), (0, pad))).reshape(n, -1, stride).max(axis=2)
    return m[:, :steps]


def _trace(model: Classifier, x: np.ndarray, mask: np.ndarray, pct: float) -> tuple[dict, dict]:
    """Run the folded float graph and record the activation range at every requantisation point."""
    enc = model.encoder
    folded = {"stem": fold(enc.stem)}
    for i, blk in enumerate(enc.blocks):
        folded[f"b{i}.c1"] = fold(blk.c1)
        folded[f"b{i}.c2"] = fold(blk.c2)
        if blk.short is not None:
            folded[f"b{i}.short"] = fold(blk.short)
    r = {"input": np.percentile(np.abs(x), pct)}
    h = np.maximum(_conv_f(x.astype(np.float64), *folded["stem"]), 0)
    r["stem"] = np.percentile(h, pct)
    for i, blk in enumerate(enc.blocks):
        a = np.maximum(_conv_f(h, *folded[f"b{i}.c1"]), 0)
        r[f"b{i}.c1"] = np.percentile(a, pct)
        c2 = _conv_f(a, *folded[f"b{i}.c2"])
        sc = _conv_f(h, *folded[f"b{i}.short"]) if blk.short is not None else h
        h = np.maximum(c2 + sc, 0)
        r[f"b{i}.out"] = np.percentile(h, pct)
    m = _mask_ds(mask.astype(np.float64), enc.total_stride, h.shape[2])[:, None, :]
    mean = (h * m).sum(2) / np.maximum(m.sum(2), 1)
    mx = np.where(m > 0, h, -np.inf).max(2)
    pooled = np.concatenate([mean, np.where(np.isfinite(mx), mx, 0)], axis=1)
    z = pooled @ enc.proj.weight.detach().double().numpy().T + enc.proj.bias.detach().double().numpy()
    r["embed"] = np.percentile(np.abs(z), pct)
    r["_z"] = z  # type: ignore[assignment]
    return r, folded


def calibrate(model: Classifier, x: np.ndarray, mask: np.ndarray, pct: float = 99.99) -> tuple[dict, dict]:
    r, folded = _trace(model, x, mask, pct)
    r.pop("_z")
    return {k: float(max(v, 1e-8)) for k, v in r.items()}, folded


def forward_folded(model: Classifier, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Float64 logits of the BN-folded graph (should equal the PyTorch model in eval mode)."""
    r, _ = _trace(model, x, mask, 50)
    return r["_z"] @ model.head.weight.detach().double().numpy().T + model.head.bias.detach().double().numpy()


# --- quantisation -----------------------------------------------------------------------------------


def _qconv(w: np.ndarray, b: np.ndarray, stride: int, s_in: float, s_out: float) -> QConv:
    sw = np.maximum(np.abs(w).reshape(len(w), -1).max(1), 1e-12) / QMAX
    wq = np.clip(np.round(w / sw[:, None, None]), -QMAX, QMAX).astype(np.int8)
    bq = np.round(b / (s_in * sw)).astype(np.int64)
    ms = [fixed_point(s_in * float(sw[c]) / s_out) for c in range(len(w))]
    return QConv(wq, bq.astype(np.int32), stride, np.array([m for m, _ in ms]), np.array([s for _, s in ms]))


def _qlinear(w: np.ndarray, b: np.ndarray, s_in: float, s_out: float | None) -> QLinear:
    sw = np.maximum(np.abs(w).max(1), 1e-12) / QMAX
    wq = np.clip(np.round(w / sw[:, None]), -QMAX, QMAX).astype(np.int8)
    bq = np.round(b / (s_in * sw)).astype(np.int32)
    if s_out is None:
        return QLinear(wq, bq, out_scale=(s_in * sw).astype(np.float64))
    ms = [fixed_point(s_in * float(sw[c]) / s_out) for c in range(len(w))]
    return QLinear(wq, bq, np.array([m for m, _ in ms]), np.array([s for _, s in ms]))


def quantise(model: Classifier, calib_x: np.ndarray, calib_mask: np.ndarray) -> QModel:
    model.eval()
    r, folded = calibrate(model, calib_x, calib_mask)
    sc = {k: v / QMAX for k, v in r.items()}
    enc = model.encoder
    stem = _qconv(*folded["stem"][:2], folded["stem"][2], sc["input"], sc["stem"])
    blocks = []
    s_prev = sc["stem"]
    for i, blk in enumerate(enc.blocks):
        s_out = sc[f"b{i}.out"]
        c1 = _qconv(*folded[f"b{i}.c1"][:2], folded[f"b{i}.c1"][2], s_prev, sc[f"b{i}.c1"])
        c2 = _qconv(*folded[f"b{i}.c2"][:2], 1, sc[f"b{i}.c1"], s_out)
        if blk.short is not None:
            short = _qconv(*folded[f"b{i}.short"][:2], folded[f"b{i}.short"][2], s_prev, s_out)
            blocks.append(QBlock(c1, c2, short))
        else:
            m, s = fixed_point(s_prev / s_out)
            blocks.append(QBlock(c1, c2, None, m, s))
        s_prev = s_out
    pw = enc.proj.weight.detach().double().numpy()
    pb = enc.proj.bias.detach().double().numpy()
    proj = _qlinear(pw, pb, s_prev, sc["embed"])
    head = _qlinear(
        model.head.weight.detach().double().numpy(), model.head.bias.detach().double().numpy(), sc["embed"], None
    )
    return QModel(calib_x.shape[2], np.float32(1.0 / sc["input"]), stem, blocks, enc.total_stride, proj, head, sc)


# --- integer inference ------------------------------------------------------------------------------


def _iconv(q: np.ndarray, c: QConv) -> np.ndarray:
    return _conv_core(q, c.w.astype(np.int64), c.stride) + c.b.astype(np.int64)[None, :, None]


def forward_int(qm: QModel, x: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (head int32 accumulators (N, K), embedding int8 (N, D))."""
    q = quantise_input(x, qm.in_inv_scale)
    h = np.clip(requant(_iconv(q, qm.stem), qm.stem.m[None, :, None], qm.stem.s[None, :, None]), 0, QMAX)
    for b in qm.blocks:
        a = np.clip(requant(_iconv(h, b.c1), b.c1.m[None, :, None], b.c1.s[None, :, None]), 0, QMAX)
        y = requant(_iconv(a, b.c2), b.c2.m[None, :, None], b.c2.s[None, :, None])
        if b.short is not None:
            sc = requant(_iconv(h, b.short), b.short.m[None, :, None], b.short.s[None, :, None])
        else:
            sc = requant(h, b.id_m, b.id_s)
        h = np.clip(y + sc, 0, QMAX)
    m = _mask_ds(mask.astype(np.int64), qm.total_stride, h.shape[2])[:, None, :]
    count = m.sum(2)
    total = (h * m).sum(2)
    mean = np.where(count > 0, (2 * total + count) // np.maximum(2 * count, 1), 0)
    mx = np.where(m > 0, h, 0).max(2)
    pooled = np.concatenate([mean, mx], axis=1).astype(np.int64)
    z = pooled @ qm.proj.w.astype(np.int64).T + qm.proj.b.astype(np.int64)
    z = np.clip(requant(z, qm.proj.m[None, :], qm.proj.s[None, :]), -QMAX, QMAX)
    acc = z @ qm.head.w.astype(np.int64).T + qm.head.b.astype(np.int64)
    return acc.astype(np.int32), z.astype(np.int8)


def logits_int(qm: QModel, x: np.ndarray, mask: np.ndarray, batch: int = 512) -> np.ndarray:
    out = []
    for i in range(0, len(x), batch):
        acc, _ = forward_int(qm, x[i : i + batch], mask[i : i + batch])
        out.append(acc * qm.head.out_scale[None, :])
    return np.concatenate(out)


# --- export for the Rust runtime ---------------------------------------------------------------------

MAGIC = b"SSEQ"
VERSION = 1


def _pack_conv(c: QConv) -> bytes:
    cout, cin, k = c.w.shape
    return (
        struct.pack("<4I", cin, cout, k, c.stride)
        + c.w.astype(np.int8).tobytes()
        + c.b.astype("<i4").tobytes()
        + c.m.astype("<i4").tobytes()
        + c.s.astype("<u4").tobytes()
    )


def export(qm: QModel) -> bytes:
    """Little-endian, self-describing enough for one architecture family; see edge-rs/src/model.rs."""
    out = [
        MAGIC,
        struct.pack("<5I", VERSION, qm.length, qm.head.w.shape[0], qm.total_stride, len(qm.blocks)),
        struct.pack("<f", float(qm.in_inv_scale)),
        _pack_conv(qm.stem),
    ]
    for b in qm.blocks:
        out += [_pack_conv(b.c1), _pack_conv(b.c2)]
        if b.short is not None:
            out += [struct.pack("<I", 1), _pack_conv(b.short)]
        else:
            out += [struct.pack("<I", 0), struct.pack("<iI", b.id_m, b.id_s)]
    p = qm.proj
    out += [
        struct.pack("<2I", p.w.shape[1], p.w.shape[0]),
        p.w.astype(np.int8).tobytes(),
        p.b.astype("<i4").tobytes(),
        p.m.astype("<i4").tobytes(),
        p.s.astype("<u4").tobytes(),
    ]
    h = qm.head
    out += [
        struct.pack("<2I", h.w.shape[1], h.w.shape[0]),
        h.w.astype(np.int8).tobytes(),
        h.b.astype("<i4").tobytes(),
        h.out_scale.astype("<f8").tobytes(),
    ]
    return b"".join(out)


def export_vectors(x: np.ndarray, mask: np.ndarray) -> bytes:
    n, _, length = x.shape
    return struct.pack("<2I", n, length) + x.astype("<f4").tobytes() + mask.astype(np.uint8).tobytes()


def export_expected(acc: np.ndarray, z: np.ndarray) -> bytes:
    return (
        struct.pack("<3I", acc.shape[0], acc.shape[1], z.shape[1])
        + acc.astype("<i4").tobytes()
        + z.astype(np.int8).tobytes()
    )
