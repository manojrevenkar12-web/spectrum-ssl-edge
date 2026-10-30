# Working in this repository

Fast feedback: `pytest -m "not slow"` (about a minute, includes the Rust parity test when cargo is
installed), `cd edge-rs && cargo test --release`, `cd web && npm test`. The whole pipeline:
`sse all --profile quick` (minutes, numbers meaningless) or `sse all` (the committed run, about an
hour on two cores). `ruff check src tests && ruff format --check src tests`; `cargo clippy -- -D warnings`;
`npm run lint`.

Invariants (each has a test):

- Settings live in `config/` (domains, experiment, scene, release thresholds), not in code.
- No model is selected, stopped or tuned on a test domain. The bench validation split fits temperatures
  only. The deployed model is chosen by the rule in `experiment.py` over dev domains.
- No pretraining pool contains the unseen symbol rates or the held-out domain.
- The Rust runtime equals the numpy int8 reference bit for bit. Change one, change the other, regenerate
  the fixtures with `sse edge`.
- `reports/results.json`, `docs/REPORT.md` and the explorer's numbers are generated. Do not edit them;
  the explorer computes nothing itself.

Held-out rule: never change anything to do better on `heldout/domain.yaml`. Its first run in
`reports/heldout/` is never rewritten. If you study its failures, write `regression` to
`heldout/STATUS` and have someone who has not seen the code write a new domain from
`heldout/WRITING_DOMAINS.md` only.
