import numpy as np
import pytest
import torch

from sse.edge import int8
from sse.models.encoder import Classifier
from sse.signals.channel import Domain
from sse.signals.dataset import make_windows
from sse.signals.modulate import KNOWN


@pytest.fixture(scope="module")
def model_and_data():
    torch.manual_seed(0)
    m = Classifier(len(KNOWN))
    # give BatchNorm non-trivial statistics, as training would
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.1, 0.1)
    m.eval()
    w = make_windows(
        KNOWN, 12, Domain("t", burst_occupancy=(0.4, 1.0), dropout=(0, 0.1), valid=(128, 256)), 256, seed=5
    )
    return m, w


@pytest.mark.parametrize("value", [1e-6, 0.0123, 0.5, 0.999999, 1.0, 3.7])
def test_fixed_point_multiplier_is_accurate(value):
    m, s = int8.fixed_point(value)
    assert (1 << 30) <= m < (1 << 31)
    assert abs(m / 2**s - value) / value < 1e-9


def test_requant_rounds_half_up_including_negatives():
    m, s = int8.fixed_point(0.5)
    out = int8.requant(np.array([3, -3, 1, -1, 4]), m, s)
    assert out.tolist() == [2, -1, 1, 0, 2]


def test_folded_float_graph_equals_pytorch(model_and_data):
    m, w = model_and_data
    with torch.no_grad():
        ref = m(torch.from_numpy(w.x), torch.from_numpy(w.mask.astype(np.float32))).double().numpy()
    got = int8.forward_folded(m, w.x, w.mask)
    assert np.max(np.abs(got - ref)) < 1e-4


def test_int8_logits_track_float_logits(model_and_data):
    m, w = model_and_data
    qm = int8.quantise(m, w.x, w.mask)
    ref = int8.forward_folded(m, w.x, w.mask)
    got = int8.logits_int(qm, w.x, w.mask)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 0.05
    assert np.mean(got.argmax(1) == ref.argmax(1)) > 0.9


def test_export_round_trip_size(model_and_data):
    m, w = model_and_data
    qm = int8.quantise(m, w.x, w.mask)
    blob = int8.export(qm)
    assert blob[:4] == b"SSEQ"
    assert qm.weight_bytes < len(blob) < qm.weight_bytes * 1.2 + 4096
