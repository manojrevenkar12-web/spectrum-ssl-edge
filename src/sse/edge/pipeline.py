"""`sse edge`: quantise the deployed model, check every implementation against every other, time it.

PyTorch float  --fold BN-->  numpy float64   (must agree to 1e-4)
numpy float    --int8-->     numpy int8 reference   (accuracy and top-1 agreement reported)
numpy int8     ==            Rust int8 runtime       (must agree bit for bit)
PyTorch float  --export-->   ONNX Runtime float      (max |diff| reported)
ONNX float     --QDQ int8--> ONNX Runtime int8       (accuracy reported, for comparison)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import config
from ..evaluate.experiment import build_data
from ..evaluate.metrics import accuracy
from ..models.encoder import Classifier
from ..signals.modulate import KNOWN
from . import int8

RUST = config.ROOT / "edge-rs"
FIXTURES = RUST / "tests" / "fixtures"


def _cargo() -> str | None:
    return shutil.which("cargo") or (
        str(Path.home() / ".cargo/bin/cargo") if (Path.home() / ".cargo/bin/cargo").exists() else None
    )


def build_rust() -> Path | None:
    cargo = _cargo()
    if cargo is None:
        return None
    subprocess.run([cargo, "build", "--release", "--quiet"], cwd=RUST, check=True)
    return RUST / "target" / "release" / "sse-edge"


def rust_infer(binary: Path, model: Path, x: np.ndarray, mask: np.ndarray, work: Path) -> tuple[np.ndarray, np.ndarray]:
    vec, out = work / "vectors.bin", work / "rust_out.bin"
    vec.write_bytes(int8.export_vectors(x, mask))
    subprocess.run([str(binary), "infer", str(model), str(vec), str(out)], check=True)
    b = out.read_bytes()
    n, k, d = np.frombuffer(b[:12], dtype="<u4")
    acc = np.frombuffer(b[12 : 12 + 4 * n * k], dtype="<i4").reshape(n, k)
    emb = np.frombuffer(b[12 + 4 * n * k :], dtype=np.int8).reshape(n, d)
    return acc, emb


def _onnx(
    model: Classifier, x: np.ndarray, mask: np.ndarray, calib: tuple[np.ndarray, np.ndarray], work: Path, tests: dict
) -> dict[str, Any]:
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    path = work / "model.onnx"
    torch.onnx.export(
        model,
        (torch.from_numpy(x[:2]), torch.from_numpy(mask[:2].astype(np.float32))),
        str(path),
        input_names=["iq", "mask"],
        output_names=["logits"],
        dynamic_axes={"iq": {0: "n"}, "mask": {0: "n"}, "logits": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    feed = {"iq": x, "mask": mask.astype(np.float32)}
    with torch.no_grad():
        ref = model(torch.from_numpy(x), torch.from_numpy(mask.astype(np.float32))).numpy()
    diff = float(np.max(np.abs(sess.run(None, feed)[0] - ref)))

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.it = iter(
                [
                    {"iq": calib[0][i : i + 32], "mask": calib[1][i : i + 32].astype(np.float32)}
                    for i in range(0, len(calib[0]), 32)
                ]
            )

        def get_next(self):  # noqa: ANN201
            return next(self.it, None)

    qpath = work / "model.int8.onnx"
    quantize_static(
        str(path),
        str(qpath),
        Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    qsess = ort.InferenceSession(str(qpath), so, providers=["CPUExecutionProvider"])
    acc_q = {
        d: accuracy(qsess.run(None, {"iq": w.x, "mask": w.mask.astype(np.float32)})[0], w.y) for d, w in tests.items()
    }

    def latency(s: Any) -> float:
        one = {"iq": x[:1], "mask": mask[:1].astype(np.float32)}
        for _ in range(50):
            s.run(None, one)
        ts = []
        for _ in range(500):
            t = time.perf_counter()
            s.run(None, one)
            ts.append(time.perf_counter() - t)
        return float(np.median(ts) * 1e6)

    return {
        "max_abs_diff_vs_pytorch": diff,
        "int8_qdq_accuracy": acc_q,
        "latency_us": {"float": latency(sess), "int8_qdq": latency(qsess)},
        "bytes": {"float": path.stat().st_size, "int8_qdq": qpath.stat().st_size},
    }


def run_edge(profile: str = "full", log=print) -> dict[str, Any]:
    run_dir = config.ROOT / "runs" / profile
    work = run_dir / "edge"
    work.mkdir(parents=True, exist_ok=True)
    ecfg = config.load("experiment.yaml")
    model = Classifier(len(KNOWN))
    model.load_state_dict(torch.load(run_dir / "models" / "deployed.pt"))
    model.eval()
    cal = np.load(run_dir / "calibration.npz")
    log("[edge] rebuilding test sets")
    data = build_data(ecfg["profiles"][profile], ecfg["length"])
    tests = data["test"]

    log("[edge] fold and check against PyTorch")
    lab = tests["lab"]
    with torch.no_grad():
        ref = model(torch.from_numpy(lab.x[:256]), torch.from_numpy(lab.mask[:256].astype(np.float32))).double().numpy()
    fold_diff = float(np.max(np.abs(int8.forward_folded(model, lab.x[:256], lab.mask[:256]) - ref)))

    log("[edge] quantise")
    qm = int8.quantise(model, cal["x"], cal["mask"])
    blob = int8.export(qm)
    (work / "model.bin").write_bytes(blob)

    per_domain: dict[str, Any] = {}
    for d, w in tests.items():
        with torch.no_grad():
            fl = model(torch.from_numpy(w.x), torch.from_numpy(w.mask.astype(np.float32))).numpy()
        q = int8.logits_int(qm, w.x, w.mask)
        per_domain[d] = {
            "float": accuracy(fl, w.y),
            "int8": accuracy(q, w.y),
            "agreement": float(np.mean(fl.argmax(1) == q.argmax(1))),
        }

    # parity set: every domain, so masks, bursts and odd symbol rates are all exercised
    take = [w.subset(np.arange(min(200, len(w)))) for w in tests.values()]
    px = np.concatenate([t.x for t in take])
    pm = np.concatenate([t.mask for t in take])
    ref_acc, ref_emb = int8.forward_int(qm, px, pm)

    # small committed fixture so `cargo test` checks parity without Python (full profile only)
    if profile == "full":
        FIXTURES.mkdir(parents=True, exist_ok=True)
        fx = np.concatenate([t.x[:8] for t in take])
        fm = np.concatenate([t.mask[:8] for t in take])
        fa, fe = int8.forward_int(qm, fx, fm)
        (FIXTURES / "model.bin").write_bytes(blob)
        (FIXTURES / "vectors.bin").write_bytes(int8.export_vectors(fx, fm))
        (FIXTURES / "expected.bin").write_bytes(int8.export_expected(fa, fe))

        _scene_fixture()

    rust: dict[str, Any] = {"available": False}
    binary = build_rust()
    if binary is not None:
        log("[edge] Rust parity and latency")
        r_acc, r_emb = rust_infer(binary, work / "model.bin", px, pm, work)
        mism = int(np.sum(np.any(r_acc != ref_acc, axis=1) | np.any(r_emb != ref_emb, axis=1)))
        info = json.loads(
            subprocess.run(
                [str(binary), "info", str(work / "model.bin")], check=True, capture_output=True, text=True
            ).stdout
        )
        bench = json.loads(
            subprocess.run(
                [str(binary), "bench", str(work / "model.bin"), str(work / "vectors.bin"), "3000"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        rust = {
            "available": True,
            "windows": int(len(px)),
            "mismatched_windows": mism,
            "bit_exact": mism == 0,
            **info,
            "latency": bench,
        }

    log("[edge] wideband scenes with the int8 model")
    from ..detect.scene import run_scenes
    from ..evaluate.metrics import Mahalanobis
    from ..signals.modulate import NOVEL

    # novelty from the int8 embedding: Mahalanobis was the best far-novel score in the evaluation (Q4)
    tr = data["train"]
    z_train = np.concatenate(
        [int8.forward_int(qm, tr.x[i : i + 512], tr.mask[i : i + 512])[1] for i in range(0, len(tr), 512)]
    )
    maha = Mahalanobis(z_train.astype(np.float64), tr.y, len(KNOWN))

    def infer(wx: np.ndarray, wm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        acc, z = int8.forward_int(qm, wx, wm)
        return acc * qm.head.out_scale[None, :], z.astype(np.float64)

    scenes = run_scenes(
        {**config.load("scene.yaml"), "scenes": ecfg["profiles"][profile]["scenes"]},
        config.domains(),
        NOVEL,
        infer,
        maha.score,
        ecfg["length"],
    )

    log("[edge] ONNX")
    onnx = _onnx(model, lab.x[:512], lab.mask[:512], (cal["x"], cal["mask"]), work, tests)

    out = {
        "deployed": "runs/" + profile + "/models/deployed.pt",
        "fold_max_abs_diff": fold_diff,
        "weight_bytes": qm.weight_bytes,
        "file_bytes": len(blob),
        "scales": qm.scales,
        "accuracy": per_domain,
        "rust": rust,
        "onnx": onnx,
        "scenes": scenes,
        "cpu": _cpu(),
    }
    with open(run_dir / "edge.json", "w") as f:
        json.dump(out, f, indent=1)
    log(
        f"[edge] int8 lab accuracy {per_domain['lab']['int8']:.3f} (float {per_domain['lab']['float']:.3f}); rust bit-exact: {rust.get('bit_exact')}"
    )
    return out


def _scene_fixture() -> None:
    """A short scene and its channel energy, for edge-rs/tests/dsp.rs."""
    import struct

    from ..detect.scene import SceneSpec, channel_energy, make_scene, spectrogram
    from ..signals.modulate import NOVEL

    spec = SceneSpec(n_samples=2**14, min_burst=2048)
    x, _ = make_scene(spec, KNOWN, NOVEL, 0.25, 7, config.domains()["lab"])
    edge_bins = config.load("scene.yaml")["detector"]["edge_bins"]
    e = channel_energy(spectrogram(x, spec.nfft, spec.hop), spec, edge_bins)
    iq = np.stack([x.real, x.imag], axis=1).astype("<f8")
    (FIXTURES / "scene_iq.bin").write_bytes(struct.pack("<I", len(x)) + iq.tobytes())
    head = struct.pack("<5I", e.shape[0], spec.channels, spec.nfft, spec.hop, edge_bins)
    (FIXTURES / "scene_energy.bin").write_bytes(head + e.astype("<f8").tobytes())


def _cpu() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"
