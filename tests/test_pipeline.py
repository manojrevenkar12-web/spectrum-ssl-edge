"""End to end on the quick profile: data, training, evaluation, edge, report. Numbers are not checked here
(the quick profile trains for seconds); the shape of the results and the edge parity are."""

import json

import pytest

from sse import config
from sse.cli import main

pytestmark = pytest.mark.slow


def test_quick_profile_end_to_end():
    assert main(["run", "--profile", "quick"]) == 0
    assert main(["edge", "--profile", "quick"]) == 0
    assert main(["report", "--profile", "quick"]) == 0
    r = json.loads((config.ROOT / "runs" / "quick" / "results.json").read_text())
    assert set(r["label_efficiency"]) == {
        "supervised",
        "ssl-lab-probe",
        "ssl-lab-finetune",
        "ssl-field-probe",
        "ssl-field-finetune",
    }
    assert "heldout" in r["domains"]
    assert r["edge"]["rust"]["bit_exact"] is True
    assert r["edge"]["onnx"]["max_abs_diff_vs_pytorch"] < 1e-3
    assert r["edge"]["scenes"]["domains"]
    assert (config.ROOT / "runs" / "quick" / "REPORT.md").read_text().startswith("# Report")
    # a quick run never writes the held-out first run
    first = config.ROOT / "reports" / "heldout" / "first_run.json"
    if first.exists():
        assert json.loads(first.read_text())["recorded_at"] != r["generated_at"]
