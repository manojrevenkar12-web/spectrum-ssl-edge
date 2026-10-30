# 3. Integer-only int8 inference, bit-exact in Rust

Status: accepted

## Context

An embedded target may have no FPU worth using, and "the model on the device behaves like the model we
evaluated" has to be checkable. Float runtimes can differ in the last bits for legitimate reasons
(fused multiply-add, summation order), which makes disagreements hard to triage.

## Decision

Per-channel symmetric int8 weights, per-tensor symmetric int8 activations calibrated on bench data,
int32 accumulation, and every rescale as `(acc * m + 2^(s-1)) >> s` in 64-bit integers. Residual
branches are requantised to the block's output scale as int32 before the add, so nothing saturates
early. Masked mean pooling divides with round-half-up. The only float operation is input quantisation
`floor(x * inv_scale + 0.5)` in float32, identical in numpy and Rust.

The numpy implementation (`src/sse/edge/int8.py`) is the reference. The Rust runtime (`edge-rs`) has no
dependencies, allocates its buffers once, and must equal the reference bit for bit: on committed
fixtures (`cargo test`), on a fresh random model with bursts, dropouts and odd symbol rates
(`tests/test_edge_parity.py`), and on every test domain in every `sse edge` run. ONNX Runtime is run
alongside, float and QDQ int8, as an independent check and a baseline.

## Consequences

- A mismatch is a bug in one of two short programs, never "numerical noise".
- The architecture is limited to what the runtime implements; adding a layer type means adding it in
  both places, and the parity tests fail until then.
- Latency on the CI CPU compares implementations, not target hardware.
