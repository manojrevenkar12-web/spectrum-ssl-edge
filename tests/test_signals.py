import numpy as np
import pytest

from sse.signals.channel import Domain, impair, multipath, power_amplifier
from sse.signals.dataset import make_windows
from sse.signals.modulate import KNOWN, NOVEL, _constellation, generate, rrc_taps


def test_rrc_has_unit_energy_and_is_symmetric():
    h = rrc_taps(8)
    assert np.isclose(np.sum(h**2), 1.0)
    assert np.allclose(h, h[::-1])


def test_rrc_cascade_is_nyquist():
    """RRC * RRC = raised cosine: zero inter-symbol interference at symbol instants."""
    sps = 8
    rc = np.convolve(rrc_taps(sps), rrc_taps(sps))
    mid = len(rc) // 2
    others = rc[mid % sps :: sps]
    peak = rc[mid]
    assert np.max(np.abs(np.delete(others, mid // sps))) < 0.02 * peak  # truncated to 8 symbols: about 1% residual ISI


@pytest.mark.parametrize("name", ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "PAM4", "32QAM"])
def test_constellations_have_unit_power_and_right_size(name):
    c = _constellation(name)
    assert np.isclose(np.mean(np.abs(c) ** 2), 1.0)
    assert len(c) == {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 16, "64QAM": 64, "PAM4": 4, "32QAM": 32}[name]


@pytest.mark.parametrize("name", KNOWN + NOVEL)
def test_every_generator_gives_unit_power_finite_windows(name):
    x = generate(name, 16, 256, 8, np.random.default_rng(0))
    assert x.shape == (16, 256)
    assert np.isfinite(x).all()
    assert np.allclose(np.mean(np.abs(x) ** 2, axis=1), 1.0, atol=1e-6)


def test_constant_envelope_waveforms_are_constant_envelope():
    for name in ["GFSK", "WBFM", "CSS", "CW"]:
        x = generate(name, 4, 256, 8, np.random.default_rng(1))
        assert np.allclose(np.abs(x), 1.0, atol=1e-9), name


def test_oqpsk_differs_from_qpsk_in_envelope():
    """Offsetting the Q rail removes zero crossings: OQPSK has a smaller envelope spread."""
    rng = np.random.default_rng(2)
    q = generate("QPSK", 200, 256, 8, rng)
    o = generate("OQPSK", 200, 256, 8, rng)
    assert np.min(np.abs(o)) > np.min(np.abs(q))


def test_measured_snr_matches_requested():
    dom = Domain(
        "t", snr_db=(10.0, 10.0), cfo=(0, 0), phase_noise=(0, 0), iq_amp_db=(0, 0), iq_phase_deg=(0, 0), dc=(0, 0)
    )
    rng = np.random.default_rng(3)
    clean = generate("QPSK", 400, 256, 8, rng)
    rx, mask, meta = impair(clean, dom, rng)
    # undo the AGC: noise power = total - 1 in units of signal power
    total = np.mean(np.abs(rx) ** 2)
    sig_share = 1 / (1 + 10 ** (-1.0))
    assert np.isclose(total, 1.0)
    # projection of rx on the clean signal (phase known) recovers the signal share of the power
    rot = np.exp(-1j * np.angle(np.sum(rx * np.conj(clean), axis=1)))[:, None]
    est = np.mean(np.abs(np.sum((rx * rot) * np.conj(clean), axis=1) / 256) ** 2)
    assert abs(est - sig_share) < 0.02


def test_cfo_shows_up_as_a_spectral_shift():
    dom = Domain(
        "t", snr_db=(40, 40), cfo=(0.05, 0.05), phase_noise=(0, 0), iq_amp_db=(0, 0), iq_phase_deg=(0, 0), dc=(0, 0)
    )
    rng = np.random.default_rng(4)
    rx, _, _ = impair(generate("CW", 1, 256, 8, np.random.default_rng(5)) * 0 + 1, dom, rng)
    peak = np.fft.fftfreq(256)[np.argmax(np.abs(np.fft.fft(rx[0])))]
    assert abs(peak - 0.05) < 1 / 256


def test_power_amplifier_compresses_peaks_not_phase():
    x = generate("64QAM", 8, 256, 8, np.random.default_rng(6))
    y = power_amplifier(x, np.full(8, 1.0), 2.0)
    papr = lambda z: np.max(np.abs(z) ** 2, axis=1) / np.mean(np.abs(z) ** 2, axis=1)  # noqa: E731
    assert np.all(papr(y) < papr(x))
    assert np.allclose(np.angle(y), np.angle(x), atol=1e-9)


def test_multipath_preserves_power_on_average():
    rng = np.random.default_rng(7)
    x = generate("QPSK", 2000, 256, 8, rng)
    y = multipath(x, np.full(2000, 4), np.full(2000, 2.0), np.full(2000, 0.0), rng)
    assert abs(np.mean(np.abs(y[:, 8:]) ** 2) - 1.0) < 0.05


def test_irregular_domain_masks_dropped_and_missing_samples():
    dom = Domain("t", burst_occupancy=(0.3, 0.5), dropout=(0.1, 0.1), valid=(100, 100))
    rng = np.random.default_rng(8)
    rx, mask, meta = impair(generate("BPSK", 50, 256, 8, rng), dom, rng)
    assert not mask[:, 100:].any()
    assert np.all(rx[~mask] == 0)
    assert np.all(mask.sum(1) <= 100)
    assert np.all(mask.sum(1) >= 100 - 0.1 * 256 - 4)


def test_windows_are_deterministic_and_balanced():
    dom = Domain("lab")
    a = make_windows(KNOWN, 5, dom, 128, seed=11)
    b = make_windows(KNOWN, 5, dom, 128, seed=11)
    assert np.array_equal(a.x, b.x)
    assert np.bincount(a.y).tolist() == [5] * len(KNOWN)
    assert make_windows(NOVEL, 3, dom, 128, seed=1).y.tolist() == [-1] * 12


def test_domain_rejects_unknown_keys_and_out_of_range_values():
    with pytest.raises(ValueError, match="unknown"):
        Domain.from_dict("x", {"snr": [0, 1]})
    with pytest.raises(ValueError, match="outside"):
        Domain.from_dict("x", {"cfo": [-0.5, 0.5]})
    with pytest.raises(ValueError, match="low, high"):
        Domain.from_dict("x", {"snr_db": [10, 0]})
