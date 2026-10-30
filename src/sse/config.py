"""Settings live in config/*.yaml; code reads them here."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .signals.channel import Domain

ROOT = Path(os.environ.get("SSE_ROOT", Path(__file__).resolve().parents[2]))


def load(name: str) -> dict[str, Any]:
    with open(ROOT / "config" / name) as f:
        return yaml.safe_load(f)


@cache
def domains() -> dict[str, Domain]:
    return {k: Domain.from_dict(k, v) for k, v in load("domains.yaml").items()}


def heldout_domain() -> Domain | None:
    p = ROOT / "heldout" / "domain.yaml"
    if not p.exists():
        return None
    with open(p) as f:
        d = yaml.safe_load(f)
    return Domain.from_dict("heldout", d["domain"])
