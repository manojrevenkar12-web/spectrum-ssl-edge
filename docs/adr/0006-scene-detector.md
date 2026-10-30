# 6. Burst detection by channel energy with hysteresis

Status: accepted

## Context

The first detector was a cell-averaging CFAR along frequency. On wideband and spread waveforms (OFDM,
chirp spread spectrum, wideband FM) the training cells fell inside the signal, the cell under test was
compared with itself, and one burst came out as up to a dozen fragments (16 detections for 7
emitters in one scene).

## Decision

Threshold the mean power in each channel, in units of the band's noise floor (median power / ln 2),
averaged over three frames. The threshold comes from a false-alarm probability under a Gamma model of
the averaged bins. A burst starts above a strict threshold (Pfa 1e-6) and lasts while above a loose one
(Pfa 1e-2), with short holes filled. The same front end is in Rust (`edge-rs/src/dsp.rs`) and matches
numpy to 1e-9 on a committed scene.

## Consequences

The floor estimate assumes emitters occupy well under half of the time-frequency plane; a crowded
band needs a floor from known-quiet channels instead. Two emitters in one channel at once are one
detection.
