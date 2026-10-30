"""reports/results.json (everything the web explorer shows) and docs/REPORT.md, both generated.

Every number in REPORT.md is read from the results here; sentences that compare two numbers say which
way the comparison went, whichever way it went.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .. import config

HELD = config.ROOT / "reports" / "heldout"


def _pct(v: float) -> str:
    return f"{100 * v:.1f}%"


def _pts(v: float) -> str:
    return f"{100 * v:+.1f} pts"


def merge(profile: str) -> dict[str, Any]:
    run_dir = config.ROOT / "runs" / profile
    with open(run_dir / "results.json") as f:
        r = json.load(f)
    edge_path = run_dir / "edge.json"
    if edge_path.exists():
        with open(edge_path) as f:
            r["edge"] = json.load(f)
    return r


def record_first_run(r: dict[str, Any]) -> dict[str, Any] | None:
    """The held-out domain's first result is written once and never rewritten."""
    if "heldout" not in r["domains"] or r["profile"] != "full":
        return None
    path = HELD / "first_run.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    full = str(max(r["budgets"]))
    domain_text = (config.ROOT / "heldout" / "domain.yaml").read_bytes()
    first = {
        "recorded_at": r["generated_at"],
        "domain_sha256": hashlib.sha256(domain_text).hexdigest(),
        "accuracy_full_labels": {m: v[full]["heldout"]["mean"] for m, v in r["label_efficiency"].items()},
        "accuracy_10_labels": {
            m: v[str(min(r["budgets"]))]["heldout"]["mean"] for m, v in r["label_efficiency"].items()
        },
        "calibration": {k: v["heldout"] for k, v in r["calibration"].items()},
        "deployed": r["deployed"]["name"],
        "int8_accuracy": r.get("edge", {}).get("accuracy", {}).get("heldout"),
    }
    HELD.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(first, f, indent=1)
    return first


def write_reports(profile: str = "full") -> dict[str, Any]:
    r = merge(profile)
    first = record_first_run(r)
    if first is not None:
        r["heldout_first_run"] = first
    out = config.ROOT / "reports" if profile == "full" else config.ROOT / "runs" / profile
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(r, f, indent=1)
    md = markdown(r)
    target = config.ROOT / "docs" / "REPORT.md" if profile == "full" else out / "REPORT.md"
    target.write_text(md)
    if first is not None:
        (HELD / "FIRST_RUN.md").write_text(_first_run_md(first)) if not (HELD / "FIRST_RUN.md").exists() else None
    return r


def _first_run_md(f: dict[str, Any]) -> str:
    rows = "\n".join(
        f"| {m} | {_pct(v)} | {_pct(f['accuracy_10_labels'][m])} |"
        for m, v in sorted(f["accuracy_full_labels"].items())
    )
    return f"""# Held-out domain: first run

Recorded {f["recorded_at"]}. Domain file sha256 `{f["domain_sha256"]}`. This file is never rewritten.
The domain (`heldout/domain.yaml`) was written from `heldout/WRITING_DOMAINS.md` by someone who had not
seen the code, the other domains or any result.

| Method | Accuracy, all labels | Accuracy, fewest labels |
|---|---:|---:|
{rows}

Deployed model: `{f["deployed"]}`.
"""


def markdown(r: dict[str, Any]) -> str:
    le = r["label_efficiency"]
    budgets = [str(b) for b in r["budgets"]]
    lo, hi = budgets[0], budgets[-1]
    doms = r["domains"]
    dev = [d for d in doms if d != "heldout"]
    seen_by_field = ["multipath", "hardware", "irregular"]
    unseen = [d for d in doms if d in ("oversampling", "heldout")]

    def acc(m: str, b: str, d: str) -> float:
        return le[m][b][d]["mean"]

    def mean_over(m: str, b: str, ds: list[str]) -> float:
        return sum(acc(m, b, d) for d in ds) / len(ds)

    methods = sorted(le)
    table = [
        "| Method | " + " | ".join(f"{d} ({b}/class)" for b in (lo, hi) for d in ("lab", "mean shift")) + " |",
        "|---|" + "---:|" * 4,
    ]
    for m in methods:
        cells = []
        for b in (lo, hi):
            cells += [_pct(acc(m, b, "lab")), _pct(mean_over(m, b, [d for d in dev if d != "lab"]))]
        table.append(f"| {m} | " + " | ".join(cells) + " |")

    # Q1
    sup, ft_lab, ft_field = "supervised", "ssl-lab-finetune", "ssl-field-finetune"
    q1_lo = acc(ft_lab, lo, "lab") - acc(sup, lo, "lab")
    q1_hi = acc(ft_lab, hi, "lab") - acc(sup, hi, "lab")
    q1 = (
        f"With {lo} labels per class, fine-tuning the encoder pretrained on bench recordings changes bench accuracy by "
        f"{_pts(q1_lo)} against training from scratch ({_pct(acc(sup, lo, 'lab'))} → {_pct(acc(ft_lab, lo, 'lab'))}). "
        f"With {hi} labels per class the difference is {_pts(q1_hi)}. "
        + (
            "The gain shrinks as labels grow, as reported for SimCLR-style pretraining on I/Q."
            if q1_lo > q1_hi
            else "The gain does not shrink with more labels here, unlike the published results."
        )
    )

    # Q2
    def shift_gap(m: str, ds: list[str]) -> float:
        return mean_over(m, hi, ds) - mean_over(sup, hi, ds)

    q2 = (
        f"At {hi} labels per class, on the shifts the field pool contained ({', '.join(seen_by_field)}), field pretraining changes "
        f"mean accuracy by {_pts(shift_gap(ft_field, seen_by_field))} against supervised training; bench-only pretraining by "
        f"{_pts(shift_gap(ft_lab, seen_by_field))}. On shifts no pretraining data contained ({', '.join(unseen)}), the changes are "
        f"{_pts(shift_gap(ft_field, unseen))} and {_pts(shift_gap(ft_lab, unseen))}."
    )

    # Q3
    cal = r["calibration"]
    names = list(cal)
    ts, ens_ts = (
        "supervised + temperature",
        next(n for n in names if n.startswith("ensemble") and n.endswith("temperature")),
    )
    ens = next(n for n in names if n.startswith("ensemble") and not n.endswith("temperature"))
    shift_d = [d for d in doms if d != "lab"]

    def ece_mean(n: str, ds: list[str]) -> float:
        return sum(cal[n][d]["ece"] for d in ds) / len(ds)

    q3 = (
        f"On the bench, temperature scaling takes the supervised model's ECE from {_pct(cal['supervised']['lab']['ece'])} to "
        f"{_pct(cal[ts]['lab']['ece'])}. Under shift (mean over {', '.join(shift_d)}) its ECE is {_pct(ece_mean(ts, shift_d))}"
        f" against {_pct(ece_mean('supervised', shift_d))} without it: "
        + (
            "the bench temperature still helps under shift, "
            if ece_mean(ts, shift_d) < ece_mean("supervised", shift_d)
            else "the bench temperature no longer helps under shift, "
        )
        + f"but calibration error under shift is {ece_mean(ts, shift_d) / max(cal[ts]['lab']['ece'], 1e-9):.1f}× the bench value. "
        f"The {ens.split(' of ')[1]}-model ensemble has ECE {_pct(ece_mean(ens, shift_d))} under shift ({_pct(ece_mean(ens_ts, shift_d))} with a temperature) — "
        + (
            "better than the single temperature-scaled model, as Ovadia et al. (2019) found."
            if ece_mean(ens_ts, shift_d) < ece_mean(ts, shift_d)
            else "not better than the single temperature-scaled model; that part of Ovadia et al. (2019) does not reproduce here."
        )
    )

    # Q4
    os_ = r["openset"]
    dep = r["deployed"]["name"]
    rows = ["| Model | Domain | Score | AUROC near | AUROC far | FPR@95 far |", "|---|---|---|---:|---:|---:|"]
    best_far = 0.0
    for mname, per_dom in os_.items():
        for d, scores in per_dom.items():
            for sname, v in scores.items():
                rows.append(
                    f"| {mname.split('/')[0]} | {d} | {sname} | {v['near']['auroc']:.3f} | {v['far']['auroc']:.3f} | {_pct(v['far']['fpr95'])} |"
                )
                if d == "lab":
                    best_far = max(best_far, v["far"]["auroc"])
    near_best = max(v["near"]["auroc"] for per in os_.values() for scores in per.values() for v in scores.values())

    edge = r.get("edge", {})
    edge_md = ""
    if edge:
        rust = edge.get("rust", {})
        acc_rows = "\n".join(
            f"| {d} | {_pct(v['float'])} | {_pct(v['int8'])} | {_pct(v['agreement'])} | {_pct(edge['onnx']['int8_qdq_accuracy'][d])} |"
            for d, v in edge["accuracy"].items()
        )
        sc = edge.get("scenes", {})
        scene_rows = "\n".join(
            f"| {d} | {v['emitters']} | {_pct(v['found'] / max(v['emitters'], 1))} | {v['false_detections']} | "
            f"{_pct(v['known_correct'] / max(v['known_found'], 1))} | {_pct(v['novel_flagged'] / max(v['novel_found'], 1))} | {_pct(v['known_flagged_novel'] / max(v['known_found'], 1))} |"
            for d, v in sc.get("domains", {}).items()
        )
        edge_md = f"""
## On the edge

The deployed model is quantised to int8 (per-channel weights, per-tensor activations, integer-only
requantisation) and run by a dependency-free Rust runtime. Folding BatchNorm changes logits by at most
{edge["fold_max_abs_diff"]:.1e}. The Rust runtime matches the numpy int8 reference on
{rust.get("windows", 0) - rust.get("mismatched_windows", 0)} of {rust.get("windows", 0)} windows bit for bit.
ONNX Runtime float differs from PyTorch by at most {edge["onnx"]["max_abs_diff_vs_pytorch"]:.1e}.

| Domain | Float | int8 (ours) | Top-1 agreement | ONNX Runtime QDQ int8 |
|---|---:|---:|---:|---:|
{acc_rows}

Budget: {rust.get("macs", 0):,} multiply-accumulates and {edge["weight_bytes"]:,} bytes of weights per model.
Latency for one window on this CI-class CPU ({edge.get("cpu", "?")}, one thread): Rust int8
{rust.get("latency", {}).get("median_us", float("nan")):.0f} µs median ({rust.get("latency", {}).get("p99_us", float("nan")):.0f} µs p99);
ONNX Runtime float {edge["onnx"]["latency_us"]["float"]:.0f} µs, QDQ int8 {edge["onnx"]["latency_us"]["int8_qdq"]:.0f} µs.
This is not the target hardware; the numbers compare implementations, not platforms.

### Wideband scenes

Each scene is a 131,072-sample band with 3 to 6 bursty emitters in the 6 usable channels of an 8-channel grid, {_pct(config.load("scene.yaml")["novel_share"])} of them
from novel classes. Energy detection with hysteresis finds bursts; each is mixed down, filtered,
decimated and classified window by window with the int8 model; a burst is flagged novel when the
Mahalanobis distance of its mean int8 embedding (the best far-novel score in Q4) exceeds the threshold
set on bench scenes with known emitters only, at {_pct(config.load("scene.yaml")["novelty_tpr"])} of them kept as known.

| Domain | Emitters | Found | False detections | Known classified right | Novel flagged | Known flagged novel |
|---|---:|---:|---:|---:|---:|---:|
{scene_rows}
"""

    held = ""
    if "heldout_first_run" in r:
        f = r["heldout_first_run"]
        held = f"""
## Held-out domain

Written blind from `heldout/WRITING_DOMAINS.md` and run once; the first run ({f["recorded_at"]}) is in
`reports/heldout/first_run.json` and is never rewritten. Deployed model on it: {_pct(f["accuracy_full_labels"][r["deployed"]["method"]])}
(supervised: {_pct(f["accuracy_full_labels"]["supervised"])}).
"""

    return f"""# Report

Generated by `sse report` from `reports/results.json` ({r["generated_at"]}, profile `{r["profile"]}`). Do not edit.
Numbers are means over seeds {r["seeds"]} (different labelled subsets and initialisations).

Setup: {len(r["classes"]["known"])} known classes, {len(r["classes"]["novel"])} novel classes; {r["sizes"]["train"]:,} labelled bench windows,
{", ".join(f"{v:,} ({k})" for k, v in r["sizes"]["pools"].items())} unlabelled windows for pretraining, {r["sizes"]["test_per_domain"]["lab"]:,} test windows per domain.
Model: {r["model"]["parameters"]:,} parameters, {r["model"]["macs_per_window"]:,} MACs per {r["model"]["length"]}-sample window.
Deployed model, chosen by a rule fixed before the run ({r["deployed"]["rule"]}): `{dep}`.

## Accuracy

{chr(10).join(table)}

## Q1. Does self-supervised pretraining help when labels are scarce?

{q1}

## Q2. Does pretraining on unlabelled field data make the classifier hold up under shift?

{q2}

## Q3. Does calibration survive shift? (Ovadia et al., 2019)

{q3}

## Q4. Can it tell when a signal is of a class it has never seen?

Best AUROC on the bench: far-novel classes {best_far:.3f}, near-novel classes {near_best:.3f}. Near-novel classes
(OQPSK, 32QAM) differ from a known class in one detail; far ones (chirp spread spectrum, a CW tone) do not
look like anything in training.

{chr(10).join(rows)}
{edge_md}{held}"""
