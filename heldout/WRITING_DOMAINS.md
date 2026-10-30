# Writing a held-out domain

You are writing one test condition for a radio-signal classifier. You must not look at the classifier's
code, its training data or its results; this brief is all you get. The domain is used once: its first
result is recorded and never rewritten, and nobody changes the classifier to do better on it.

## What the classifier does

It classifies 256-sample complex-baseband windows into ten modulation classes: BPSK, QPSK, 8PSK, 16QAM,
64QAM, PAM4 (all root-raised-cosine shaped), GFSK, AM-DSB, WBFM and a small OFDM signal. Labelled data
was recorded on a bench. It will be used on receivers in the field.

## What a domain is

A domain is a set of parameter ranges. Each window draws every parameter uniformly from its range. Write
the domain as YAML in this shape, using only these keys (omit a key to keep its default):

```yaml
domain:
  snr_db: [low, high]           # signal-to-noise ratio, dB.        supported -20..40, default [-4, 20]
  sps: [a, b, ...]              # samples per symbol, integers.     supported 2..32,   default [8]
  cfo: [low, high]              # carrier frequency offset, cycles/sample. supported -0.05..0.05, default [-0.002, 0.002]
  phase_noise: [low, high]      # random-walk phase noise, rad/sample std. supported 0..0.05, default [0, 0.002]
  iq_amp_db: [low, high]        # receiver IQ amplitude imbalance, dB. supported 0..3,  default [0, 0.5]
  iq_phase_deg: [low, high]     # receiver IQ phase imbalance, deg. supported 0..15,   default [0, 2]
  dc: [low, high]               # DC offset relative to signal RMS. supported 0..0.3,  default [0, 0.02]
  pa_backoff_db: [low, high]    # transmitter amplifier saturation above signal RMS, dB (Rapp model);
                                #   omit for a linear amplifier. supported 0..12
  multipath_taps: [low, high]   # number of channel taps (1 sample apart); omit for no multipath. supported 1..8
  multipath_decay: [low, high]  # exponential power-delay decay, samples. supported 0.2..8, default [1, 3]
  multipath_k_db: [low, high]   # Rician K-factor of the first tap, dB. supported -10..20, default [0, 6]
  burst_occupancy: [low, high]  # share of the window the emitter is on; omit for always on. supported 0.1..1
  dropout: [low, high]          # share of samples lost in 1-4 gaps. supported 0..0.3, default [0, 0]
  valid: [low, high]            # samples recorded before the capture ends; omit for all 256. supported 32..256
```

## What to write

Describe a plausible field deployment that is *not* the bench, in a sentence or two (as a YAML comment),
then the domain. Combine several effects the way a real deployment would; do not make it trivially easy
or impossible (keep the SNR range overlapping 0..20 dB). Write the file `heldout/domain.yaml`, then write
`heldout/STATUS` containing the single word `blind`.
