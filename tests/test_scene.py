import numpy as np

from sse.detect.scene import (
    SceneSpec,
    channel_energy,
    channelise,
    classify_detections,
    detect,
    energy_threshold,
    make_scene,
    score_scene,
    spectrogram,
)
from sse.signals.channel import Domain
from sse.signals.modulate import KNOWN, NOVEL

CFG = {"pfa": 1e-6, "pfa_hold": 1e-2, "edge_bins": 2, "smooth_frames": 3, "min_frames": 6, "gap_frames": 6}


def test_threshold_false_alarm_rate_in_pure_noise():
    spec = SceneSpec()
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(2**18) + 1j * rng.standard_normal(2**18)) / np.sqrt(2)
    e = channel_energy(spectrogram(x, spec.nfft, spec.hop), spec, CFG["edge_bins"])
    thr = energy_threshold(spec, CFG["edge_bins"], 1, 0.01)
    rate = np.mean(e > thr)
    assert rate < 0.03  # nominal 1%; Hann-correlated bins make the model approximate


def test_no_detections_in_noise_only():
    spec = SceneSpec()
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(spec.n_samples) + 1j * rng.standard_normal(spec.n_samples)) / np.sqrt(2)
    _, dets = detect(x, spec, CFG)
    assert dets == []


def test_every_strong_emitter_is_found_in_its_channel():
    spec = SceneSpec(snr_db=(10.0, 16.0))
    for seed in range(5):
        x, ems = make_scene(spec, KNOWN, NOVEL, 0.25, seed, Domain("lab"))
        _, dets = detect(x, spec, CFG)
        classify_detections(
            x,
            dets,
            spec,
            lambda wx, wm: (np.zeros((len(wx), len(KNOWN))), np.zeros((len(wx), 4))),
            256,
            lambda z: np.zeros(len(z)),
            np.inf,
        )
        s = score_scene(ems, dets)
        assert s["found"] == s["emitters"], (seed, s)
        assert s["false_detections"] == 0


def test_channelise_brings_the_emitter_to_baseband():
    spec = SceneSpec(min_emitters=1, max_emitters=1, snr_db=(20.0, 20.0))
    x, ems = make_scene(spec, ["CW"], [], 0.0, 3, Domain("lab", cfo=(0.0, 0.0)))
    _, dets = detect(x, spec, CFG)
    base = channelise(x, dets[0], spec)
    f = np.fft.fftfreq(len(base))[np.argmax(np.abs(np.fft.fft(base)))]
    assert abs(f) < 0.06  # the tone sits within the CW generator's +-0.05 offset of the channel centre
    assert ems[0].channel == dets[0].channel
