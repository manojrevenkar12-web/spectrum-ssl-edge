# 2. Contrastive pretraining with RF augmentations, fixed schedules

Status: accepted

## Decision

SimCLR (NT-Xent, temperature 0.2) on unlabelled windows. Two views of a window differ by nuisances the
label must not depend on: carrier phase, a small frequency offset, the window start, how much of the
burst was captured, extra noise, a dropped stretch of samples. Multipath and hardware effects are
deliberately *not* augmentations, so that "field pretraining" measures what unlabelled field data adds,
not what a hand-written augmentation adds.

Two pools: bench only (isolates self-supervision) and field (bench plus unlabelled multipath, hardware
and irregular recordings). Symbol rates the bench never recorded and the held-out domain are in no
pool, so the report can separate "saw this shift without labels" from "never saw it".

Every model follows a fixed schedule from `config/experiment.yaml`. There is no early stopping and no
model selection on any test domain; the bench validation split fits temperatures and nothing else.
The deployed model is chosen by a rule fixed before the run (best mean accuracy over the dev domains).

## Consequences

The comparison is fair but not tuned: a method that would win with its own tuned schedule may lose
here. Seeds vary both the labelled subset and the initialisation; the explorer shows their spread.
