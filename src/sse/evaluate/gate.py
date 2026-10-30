"""`sse gate`: the release check against config/release.yaml, from reports/results.json."""

from __future__ import annotations

import json
from typing import Any

from .. import config


def _load(profile: str) -> dict[str, Any]:
    path = (
        config.ROOT / "reports" / "results.json"
        if profile == "full"
        else config.ROOT / "runs" / profile / "results.json"
    )
    with open(path) as f:
        return json.load(f)


def gate(profile: str = "full") -> tuple[bool, list[str]]:
    r = _load(profile)
    rel = config.load("release.yaml")
    t = rel["floors"]
    full = str(max(r["budgets"]))
    dep_method = r["deployed"]["method"]
    lab_acc = r["label_efficiency"][dep_method][full]["lab"]["mean"]
    hi_rows = [b for b in r["accuracy_by_snr"][dep_method]["lab"] if b["lo"] >= 12]
    hi_acc = sum(b["accuracy"] * b["n"] for b in hi_rows) / max(1, sum(b["n"] for b in hi_rows))
    cal_name = f"{dep_method} + temperature"
    ece_lab = r["calibration"][cal_name]["lab"]["ece"]
    far = max(v["far"]["auroc"] for scores in r["openset"].values() for v in scores["lab"].values())
    e = r.get("edge", {})
    rust = e.get("rust", {})
    checks = [
        ("bench accuracy (deployed)", lab_acc, ">=", t["lab_accuracy_min"]),
        ("bench accuracy SNR >= 12 dB", hi_acc, ">=", t["lab_accuracy_high_snr_min"]),
        ("bench ECE after temperature", ece_lab, "<=", t["ece_lab_max"]),
        ("far-novel AUROC (best score)", far, ">=", t["far_ood_auroc_min"]),
    ]
    if e:
        checks += [
            ("int8 top-1 agreement (bench)", e["accuracy"]["lab"]["agreement"], ">=", t["int8_agreement_min"]),
            ("ONNX max |diff| vs PyTorch", e["onnx"]["max_abs_diff_vs_pytorch"], "<=", t["onnx_max_abs_diff"]),
            ("Rust bit-exact with reference", float(bool(rust.get("bit_exact"))), ">=", float(t["rust_bit_exact"])),
            ("MACs per window", float(rust.get("macs", e.get("macs", 0))), "<=", t["edge_macs_max"]),
            ("weight bytes", float(e["weight_bytes"]), "<=", t["edge_weights_bytes_max"]),
        ]
    else:
        checks.append(("edge results present", 0.0, ">=", 1.0))
    ok = True
    lines = []
    for name, v, op, thr in checks:
        good = v >= thr if op == ">=" else v <= thr
        ok &= good
        lines.append(f"{'pass' if good else 'FAIL'}  {name:34s} {v:.4g} ({op} {thr})")
    targets = rel.get("targets", {})
    for name, v, key in [
        ("target: bench accuracy SNR >= 12 dB", hi_acc, "lab_accuracy_high_snr"),
        ("target: far-novel AUROC", far, "far_ood_auroc"),
    ]:
        if key in targets:
            met = v >= targets[key]
            lines.append(f"{'met ' if met else 'miss'}  {name:34s} {v:.4g} (>= {targets[key]}, reported, not gated)")
    return ok, lines
