"""The experiment's rules: no test domain in pretraining, the held-out domain untouched after its first run."""

import hashlib
import json

from sse import config


def test_pretraining_pools_never_contain_the_unseen_shifts():
    pools = config.load("experiment.yaml")["ssl_pools"]
    for name, sources in pools.items():
        assert "oversampling" not in sources, name
        assert "heldout" not in sources, name


def test_heldout_domain_is_valid_and_blind():
    assert (config.ROOT / "heldout" / "STATUS").read_text().strip() in {"blind", "regression"}
    dom = config.heldout_domain()
    assert dom is not None


def test_heldout_domain_unchanged_since_its_first_run():
    first = config.ROOT / "reports" / "heldout" / "first_run.json"
    text = (config.ROOT / "heldout" / "domain.yaml").read_bytes()
    recorded = (config.ROOT / "heldout" / "domain.sha256").read_text().split()[0]
    assert hashlib.sha256(text).hexdigest() == recorded
    if first.exists():
        assert json.loads(first.read_text())["domain_sha256"] == recorded


def test_model_selection_rule_ignores_the_heldout_domain():
    src = (config.ROOT / "src" / "sse" / "evaluate" / "experiment.py").read_text()
    assert 'dev_domains = [d for d in domains if d != "heldout"]' in src
    assert "max(methods, key=lambda m: dev_mean(m, full))" in src
