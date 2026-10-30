# 4. A held-out domain written blind, run once

Status: accepted

The dev domains were written by the person who wrote the model, so robustness on them is partly
designed in. `heldout/domain.yaml` was written from `heldout/WRITING_DOMAINS.md` alone by someone who
had not seen the code, the dev domains or any result; its sha256 is in `heldout/domain.sha256`.

Rules (`tests/test_protocol.py` checks the mechanical ones):

- Never change code, configuration or training to do better on it.
- Its first run (`reports/heldout/first_run.json`) is written once and never rewritten.
- If anyone studies its failures, write `regression` to `heldout/STATUS`, move it into the dev
  domains, and have someone who has not seen the code write a new one from the brief.
