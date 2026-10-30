"""The Rust runtime against the numpy int8 reference, on a model and windows made here."""

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from sse.edge import int8
from sse.edge.pipeline import RUST, build_rust, rust_infer
from sse.models.encoder import Classifier
from sse.signals.channel import Domain
from sse.signals.dataset import make_windows
from sse.signals.modulate import KNOWN

pytestmark = pytest.mark.skipif(
    shutil.which("cargo") is None and not (Path.home() / ".cargo/bin/cargo").exists(),
    reason="Rust toolchain not installed",
)


def test_rust_is_bit_exact_with_the_reference(tmp_path):
    torch.manual_seed(1)
    m = Classifier(len(KNOWN))
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm1d):
            mod.running_mean.uniform_(-0.3, 0.3)
            mod.running_var.uniform_(0.3, 3.0)
    m.eval()
    parts = [
        make_windows(KNOWN, 6, Domain("a"), 256, seed=1),
        make_windows(
            KNOWN, 6, Domain("b", burst_occupancy=(0.2, 0.6), dropout=(0.0, 0.2), valid=(40, 200)), 256, seed=2
        ),
        make_windows(KNOWN, 6, Domain("c", sps=(3, 5, 13), snr_db=(-10.0, 30.0)), 256, seed=3),
    ]
    x = np.concatenate([p.x for p in parts])
    mask = np.concatenate([p.mask for p in parts])
    qm = int8.quantise(m, x, mask)
    (tmp_path / "model.bin").write_bytes(int8.export(qm))
    binary = build_rust()
    acc, emb = rust_infer(binary, tmp_path / "model.bin", x, mask, tmp_path)
    ref_acc, ref_emb = int8.forward_int(qm, x, mask)
    assert np.array_equal(acc, ref_acc)
    assert np.array_equal(emb, ref_emb)


def test_committed_fixtures_exist_for_cargo_test():
    for f in ("model.bin", "vectors.bin", "expected.bin"):
        assert (RUST / "tests" / "fixtures" / f).stat().st_size > 0
