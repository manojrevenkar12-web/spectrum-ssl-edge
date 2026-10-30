"""Wideband scenes: several bursty emitters in a band, found, channelised, classified and flagged.

This is how the classifier is used on a receiver: it does not get neat 256-sample windows, it gets a
band with emitters keying up and down. The scene generator places emitters (known and novel classes)
in a channel grid at random times; the detector thresholds channel energy in a spectrogram, turns it
into (channel, start, end) detections, mixes each down, filters, decimates to the classifier's rate,
classifies every window with the int8 model, and flags a detection as novel when the Mahalanobis
distance of its mean int8 embedding is above the threshold set on bench scenes with known emitters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import firwin, lfilter
from scipy.special import softmax

from ..signals.channel import Domain, impair
from ..signals.modulate import KNOWN, generate


@dataclass(frozen=True)
class SceneSpec:
    n_samples: int = 2**17
    channels: int = 8  # channel spacing fs / channels
    decim: int = 8  # wideband samples per classifier sample
    nfft: int = 256
    hop: int = 128
    min_emitters: int = 3
    max_emitters: int = 6
    min_burst: int = 4096  # wideband samples
    snr_db: tuple[float, float] = (-2.0, 16.0)  # per emitter, in its channel's bandwidth


@dataclass
class Emitter:
    cls: str
    channel: int
    start: int
    end: int
    snr_db: float


def make_scene(
    spec: SceneSpec, known: list[str], novel: list[str], novel_share: float, seed: int, dom: Domain
) -> tuple[np.ndarray, list[Emitter]]:
    rng = np.random.default_rng(seed)
    n = spec.n_samples
    x = np.zeros(n, dtype=np.complex128)
    k = rng.integers(spec.min_emitters, spec.max_emitters + 1)
    # channel 0 holds the receiver's DC; channel channels/2 straddles the band edge
    usable = np.array([c for c in range(1, spec.channels) if c != spec.channels // 2])
    chans = rng.choice(usable, size=min(k, len(usable)), replace=False)
    ems: list[Emitter] = []
    for ch in chans:
        cls = str(rng.choice(novel)) if (novel and rng.uniform() < novel_share) else str(rng.choice(known))
        dur = int(rng.integers(spec.min_burst, n // 2))
        start = int(rng.integers(0, n - dur))
        nb = dur // spec.decim + 64
        base = generate(cls, 1, nb, 8, rng)
        # nuisance effects of the domain, without its noise (the band has one noise floor)
        quiet = Domain(
            dom.name,
            snr_db=(80.0, 80.0),
            sps=dom.sps,
            cfo=dom.cfo,
            phase_noise=dom.phase_noise,
            iq_amp_db=(0.0, 0.0),
            iq_phase_deg=(0.0, 0.0),
            dc=(0.0, 0.0),
            pa_backoff_db=dom.pa_backoff_db,
            multipath_taps=dom.multipath_taps,
            multipath_decay=dom.multipath_decay,
            multipath_k_db=dom.multipath_k_db,
        )
        base, _, _ = impair(base, quiet, rng)
        up = np.zeros(nb * spec.decim, dtype=np.complex128)
        up[:: spec.decim] = base[0]
        interp = firwin(8 * spec.decim + 1, 1.0 / spec.decim) * spec.decim
        sig = lfilter(interp, 1.0, up)[4 * spec.decim : 4 * spec.decim + dur]
        snr = float(rng.uniform(*spec.snr_db))
        # noise in one channel has power 1/channels of the unit band noise
        amp = np.sqrt(10 ** (snr / 10) / spec.channels) / np.sqrt(np.mean(np.abs(sig) ** 2))
        f = ch / spec.channels
        t = np.arange(start, start + dur)
        x[start : start + dur] += amp * sig * np.exp(2j * np.pi * f * t)
        ems.append(Emitter(cls, int(ch), start, start + dur, snr))
    x += (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return x, ems


def spectrogram(x: np.ndarray, nfft: int, hop: int) -> np.ndarray:
    frames = (len(x) - nfft) // hop + 1
    idx = np.arange(frames)[:, None] * hop + np.arange(nfft)[None, :]
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)
    return np.abs(np.fft.fftshift(np.fft.fft(x[idx] * win, axis=1), axes=1)) ** 2


def channel_energy(power: np.ndarray, spec: SceneSpec, edge_bins: int) -> np.ndarray:
    """Mean power in each channel per frame, in units of the band's noise floor.

    The floor is estimated as median(power) / ln 2 (the median of an exponential variable is ln 2
    times its mean), which holds while emitters occupy less than about half of the time-frequency
    plane. Same definition as `channel_energy` in edge-rs/src/dsp.rs.
    """
    floor = np.median(power) / np.log(2)
    bins_per_ch = spec.nfft // spec.channels
    half = bins_per_ch // 2 - edge_bins
    out = np.zeros((power.shape[0], spec.channels))
    for c in range(spec.channels):
        f = c / spec.channels
        f = f - 1 if f >= 0.5 else f
        centre = int(round(spec.nfft / 2 + f * spec.nfft))
        out[:, c] = power[:, max(0, centre - half) : min(spec.nfft, centre + half + 1)].mean(axis=1) / floor
    return out


def energy_threshold(spec: SceneSpec, edge_bins: int, smooth: int, pfa: float) -> float:
    """Threshold on the smoothed channel energy for a per-frame false-alarm probability in noise.

    Mean of n exponential bins over `smooth` frames ~ Gamma(n_eff, 1/n_eff); a Hann window makes
    neighbouring bins correlated, so n_eff is taken as half the bins.
    """
    from scipy.special import gammaincinv

    n_bins = 2 * (spec.nfft // spec.channels // 2 - edge_bins) + 1
    n_eff = max(1.0, n_bins / 2 * smooth)
    return float(gammaincinv(n_eff, 1 - pfa) / n_eff)


@dataclass
class Detection:
    channel: int
    start: int
    end: int
    cls: str = ""
    novel: bool = False
    confidence: float = 0.0
    novelty: float = 0.0


def detect(x: np.ndarray, spec: SceneSpec, cfg: dict[str, Any]) -> tuple[np.ndarray, list[Detection]]:
    power = spectrogram(x, spec.nfft, spec.hop)
    e = channel_energy(power, spec, cfg["edge_bins"])
    k = cfg["smooth_frames"]
    if k > 1:
        e = np.apply_along_axis(lambda v: np.convolve(v, np.ones(k) / k, mode="same"), 0, e)
    # hysteresis: a burst starts above the strict threshold and lasts while above the loose one,
    # so a fading emitter is one detection, not a string of fragments
    high = e > energy_threshold(spec, cfg["edge_bins"], k, cfg["pfa"])
    low = e > energy_threshold(spec, cfg["edge_bins"], k, cfg["pfa_hold"])
    dets: list[Detection] = []
    for c in range(1, spec.channels):
        if c == spec.channels // 2:
            continue
        col = _close(low[:, c], cfg["gap_frames"])
        edges = np.flatnonzero(np.diff(np.r_[0, col.astype(int), 0]))
        for s, e_ in zip(edges[::2], edges[1::2], strict=True):
            if e_ - s >= cfg["min_frames"] and high[s:e_, c].any():
                dets.append(Detection(c, int(s * spec.hop), int(min(len(x), (e_ - 1) * spec.hop + spec.nfft))))
    return power, dets


def _close(v: np.ndarray, gap: int) -> np.ndarray:
    """Fill holes of up to `gap` frames inside a run (a burst seen through fading)."""
    out = v.copy()
    idx = np.flatnonzero(v)
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        if 1 < b - a <= gap + 1:
            out[a:b] = True
    return out


def channelise(x: np.ndarray, det: Detection, spec: SceneSpec) -> np.ndarray:
    f = det.channel / spec.channels
    seg = x[det.start : det.end] * np.exp(-2j * np.pi * f * np.arange(det.start, det.end))
    lp = firwin(16 * spec.decim + 1, 1.0 / spec.decim)
    return lfilter(lp, 1.0, seg)[:: spec.decim]


def windows_of(base: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-overlapping windows; a short tail is kept with a mask. AGC per window."""
    n = max(1, int(np.ceil(len(base) / length)))
    xw = np.zeros((n, length), dtype=np.complex128)
    mw = np.zeros((n, length), dtype=bool)
    for i in range(n):
        seg = base[i * length : (i + 1) * length]
        xw[i, : len(seg)] = seg
        mw[i, : len(seg)] = True
    if n > 1 and mw[-1].sum() < length // 2:  # drop a tiny tail
        xw, mw = xw[:-1], mw[:-1]
    p = np.sum(np.abs(xw) ** 2, axis=1) / np.maximum(mw.sum(1), 1)
    xw = xw / np.sqrt(np.maximum(p, 1e-12))[:, None]
    return np.stack([xw.real, xw.imag], axis=1).astype(np.float32), mw


def classify_detections(
    x: np.ndarray, dets: list[Detection], spec: SceneSpec, infer_fn, length: int, novelty_fn, threshold: float
) -> None:
    """infer_fn(windows, masks) -> (logits, embeddings); novelty_fn(embedding (1, D)) -> score."""
    for d in dets:
        wx, wm = windows_of(channelise(x, d, spec), length)
        lo, emb = infer_fn(wx, wm)
        mean = lo.mean(axis=0)
        p = softmax(mean)
        d.cls = KNOWN[int(np.argmax(mean))]
        d.confidence = float(p.max())
        d.novelty = float(novelty_fn(emb.mean(axis=0, keepdims=True))[0])
        d.novel = d.novelty > threshold


def score_scene(truth: list[Emitter], dets: list[Detection], min_overlap: float = 0.5) -> dict[str, Any]:
    matched: list[tuple[Emitter, Detection]] = []
    used: set[int] = set()
    for t in truth:
        best, best_ov = None, 0.0
        for i, d in enumerate(dets):
            if i in used or d.channel != t.channel:
                continue
            inter = max(0, min(t.end, d.end) - max(t.start, d.start))
            ov = inter / (t.end - t.start)
            if ov > best_ov:
                best, best_ov = i, ov
        if best is not None and best_ov >= min_overlap:
            used.add(best)
            matched.append((t, dets[best]))
    known = [(t, d) for t, d in matched if t.cls in KNOWN]
    novel = [(t, d) for t, d in matched if t.cls not in KNOWN]
    return {
        "emitters": len(truth),
        "detections": len(dets),
        "found": len(matched),
        "false_detections": len(dets) - len(used),
        "known_found": len(known),
        "known_correct": sum(1 for t, d in known if d.cls == t.cls and not d.novel),
        "known_flagged_novel": sum(1 for _, d in known if d.novel),
        "novel_found": len(novel),
        "novel_flagged": sum(1 for _, d in novel if d.novel),
    }


def run_scenes(
    cfg: dict[str, Any], doms: dict[str, Domain], novel: list[str], infer_fn, novelty_fn, length: int
) -> dict[str, Any]:
    """Calibrate the novelty threshold on bench scenes with known emitters only, then score every domain."""
    spec = SceneSpec()
    det_cfg = cfg["detector"]
    calib: list[float] = []
    for seed in range(cfg["scenes"]):
        x, _ = make_scene(spec, KNOWN, [], 0.0, 90_000 + seed, doms["lab"])
        _, dets = detect(x, spec, det_cfg)
        classify_detections(x, dets, spec, infer_fn, length, novelty_fn, np.inf)
        calib += [d.novelty for d in dets]
    threshold = float(np.quantile(calib, cfg["novelty_tpr"])) if calib else float("inf")
    out: dict[str, Any] = {
        "novelty_threshold": threshold,
        "novelty_score": "mahalanobis",
        "calibration_detections": len(calib),
        "domains": {},
        "example": None,
    }
    for j, name in enumerate(cfg["domains"]):
        tot: dict[str, int] = {}
        for seed in range(cfg["scenes"]):
            x, ems = make_scene(spec, KNOWN, novel, cfg["novel_share"], 100_000 * (j + 1) + seed, doms[name])
            power, dets = detect(x, spec, det_cfg)
            classify_detections(x, dets, spec, infer_fn, length, novelty_fn, threshold)
            for k, v in score_scene(ems, dets).items():
                tot[k] = tot.get(k, 0) + v
            if out["example"] is None and name == "lab" and len(ems) >= 5 and any(e.cls not in KNOWN for e in ems):
                out["example"] = _example(power, ems, dets, spec)
        out["domains"][name] = tot
    return out


def _example(
    power: np.ndarray, ems: list[Emitter], dets: list[Detection], spec: SceneSpec, frames: int = 256
) -> dict[str, Any]:
    """A downsampled waterfall (dB, 0-255) with truth and detections, for the results explorer."""
    step = max(1, power.shape[0] // frames)
    p = power[: step * (power.shape[0] // step)].reshape(-1, step, power.shape[1]).mean(axis=1)
    p = p.reshape(p.shape[0], -1, 2).mean(axis=2)  # 128 bins
    db = 10 * np.log10(p + 1e-12)
    lo, hi = np.percentile(db, 5), np.percentile(db, 99.9)
    img = np.clip((db - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    hop_per_row = spec.hop * step
    return {
        "rows": int(img.shape[0]),
        "cols": int(img.shape[1]),
        "channels": spec.channels,
        "pixels": img.ravel().tolist(),
        "samples_per_row": hop_per_row,
        "truth": [
            {
                "cls": e.cls,
                "channel": e.channel,
                "start": e.start / hop_per_row,
                "end": e.end / hop_per_row,
                "snr_db": round(e.snr_db, 1),
                "novel": e.cls not in KNOWN,
            }
            for e in ems
        ],
        "detections": [
            {
                "cls": d.cls,
                "channel": d.channel,
                "start": d.start / hop_per_row,
                "end": d.end / hop_per_row,
                "novel": bool(d.novel),
                "confidence": round(d.confidence, 3),
            }
            for d in dets
        ],
    }
