# ADR 422: KGE ranks proposals without evidence authority

Status: Research implementation. Predictive and field utility unproven.

Date: 2026 10 13 UTC

## Decision

Use exact pinned `@ruvector/kge` 0.2.0 in a separate optional Node experiment.
Call the real backend and expose failures rather than silently substituting a
mock. Keep installation and model training out of the Rust decision path.

The initial graph has authored context, sensor and relevance relationships.
Training excludes the held out target links. Compare trained ranking with a
simple frequency baseline and a permuted relevance control. A completed training
run, high training score or saved artifact is not evidence of unseen room utility.
The fixture is SYNTHETIC and its context entities are known to the model through
side facts. This is not an inductive unseen entity benchmark.

Use HolE for supported link ranking. Do not assert that HolE implements the
adapter's RotatE composition operation. Native and WASM timing must be labelled
separately. Model hashes identify bytes; they do not authenticate a trusted author.

## Interface

The output is bounded JSON with exactly `schemaVersion: 1`, `evidence:
"SYNTHETIC"` and `candidates`. Each candidate has an allowlisted `sensorId` and a
finite numeric `score`. No device address, command, executable path, threshold,
occupancy state or evidence root is accepted from the ranker. Candidate scores
are ordering values, not calibrated probabilities or expected information gain.
Any tie uses deterministic ordering. Invalid output fails closed.

Rust independently checks available sensors and all sensing budget dimensions.
A suggestion can be refused and cannot add independent evidence. It cannot
silently increase sensing or replace a stale observation with a plausible edge.

## Risk and alternatives

Frequency ranking is the mandatory simpler comparator. KGE costs training,
memory and inference time and may memorize familiar room IDs. Real deployment
requires an explicit new entity initialization/adaptation strategy and validation
on fresh rooms with frozen thresholds. A ranking model cannot identify causal
sensor utility from observational correlation alone. Prospective randomized
policies or complete eligible measurement replay are required.

## Acceptance

Contract tests check real training, finite allowed candidates, deterministic
ordering, persistence where supported, bounded inputs and missing dependency
failure. Record package and backend versions and training/fixture hashes.
Benchmark ranking without retraining on every request. Report all fixed controls,
including negative results. No automatic model promotion or hardware calls.

Source: https://gist.github.com/ruvnet/3148ac8628c76d098fda7293fdc4a222

The cited core SDK wrapper was documented as a source integration, not proof
that every published core SDK contains those exports. This experiment calls
the published KGE package directly.
