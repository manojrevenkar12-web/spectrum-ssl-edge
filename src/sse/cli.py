"""`sse run` (experiment), `sse edge` (int8, ONNX, Rust parity), `sse report`, `sse gate`, `sse all`."""

from __future__ import annotations

import argparse
import sys

from . import config


def _run(a: argparse.Namespace) -> int:
    from .evaluate.experiment import run

    run(a.profile)
    return 0


def _edge(a: argparse.Namespace) -> int:
    from .edge.pipeline import run_edge

    run_edge(a.profile)
    return 0


def _report(a: argparse.Namespace) -> int:
    from .evaluate.report import write_reports

    write_reports(a.profile)
    return 0


def _gate(a: argparse.Namespace) -> int:
    from .evaluate.gate import gate

    ok, lines = gate(a.profile)
    print("\n".join(lines))
    return 0 if ok else 1


def _all(a: argparse.Namespace) -> int:
    for step in (_run, _edge, _report, _gate):
        rc = step(a)
        if rc:
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sse")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in [
        ("run", _run, "build data, train every model, evaluate"),
        ("edge", _edge, "fold, quantise, export; ONNX and Rust parity; latency"),
        ("report", _report, "write reports/results.json and docs/REPORT.md"),
        ("gate", _gate, "check config/release.yaml"),
        ("all", _all, "run, edge, report, gate"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--profile", default="full", choices=sorted(config.load("experiment.yaml")["profiles"]))
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

__all__ = ["main"]
