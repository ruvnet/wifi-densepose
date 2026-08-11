# ADR-293: Vitals ground-truth rig — reference ingest, time alignment, and agreement metrics

- **Status**: Accepted — initial implementation (this PR)
- **Date**: 2026-08-10
- **Deciders**: ruv
- **Tags**: vitals, validation, ground-truth, bland-altman, evidence, honesty

## Context

`wifi-densepose-vitals` (ADR-021) extracts breathing (0.1–0.5 Hz) and heart
rate (0.8–2.0 Hz) from CSI. The 2026 research sweep found that every credible
vitals result in the literature ships with reference-sensor ground truth
(chest strap, pulse oximeter, ECG, or PSG), and that WiFi heart-rate numbers
without stated scope (single person, static, line-of-sight, short range) are
systematically misleading. RuView currently has no way to produce a MEASURED
vitals number: there is no reference-signal ingest, no time alignment between
CSI-derived estimates and a reference device, and no agreement statistics.

CLAUDE.md requires accuracy statements to be tagged MEASURED (with a
reproducer), CLAIMED, or SYNTHETIC. For vitals, MEASURED is currently
unreachable.

## Options considered

1. **Live BLE/ANT+ integration with reference devices.** Rejected for now:
   drivers and pairing are a hardware/product concern; the blocking gap is
   the evaluation math, not the radio link.
2. **File-based reference ingest + offline agreement analysis.** Chosen:
   every consumer reference device (Polar, Garmin, oximeters) exports
   timestamped series; a file boundary keeps the crate dependency-free and
   the pipeline deterministic.

## Decision

Add a `groundtruth` module to `v2/crates/wifi-densepose-vitals`:

### 1. Reference series ingest

- `ReferenceSeries`: timestamped samples (unix millis + value) for one
  measurand (`HeartRateBpm` or `BreathingRateBrpm`), with device metadata
  (make/model, measurement principle). Parsed from CSV (`timestamp_ms,value`
  with a header line); malformed rows are rejected with row-numbered errors —
  untrusted file input validated at the boundary. Non-monotonic timestamps
  are an error, not silently sorted.

### 2. Time alignment

- Constant-offset estimation by maximizing normalized cross-correlation of
  the estimate series against the reference over a bounded lag window
  (default ±30 s), on a common resampled grid (nearest-sample, no
  interpolation of physiological values across gaps larger than a
  configurable limit).
- Optional linear clock-drift fit (offset + rate) for long sessions.
  Alignment parameters are reported, never silently applied.

### 3. Agreement metrics

- `AgreementReport`: n paired samples, coverage fraction (time where both
  series had valid samples), MAE, RMSE, mean error (bias), Bland–Altman
  95% limits of agreement, and percentage-within-tolerance (configurable,
  default ±2 bpm HR / ±1 brpm breathing).
- Session scope is mandatory metadata: subject count, motion state
  (static/moving), line-of-sight (LOS/NLOS/through-wall), distance band.
  A report without scope cannot be constructed.

### 4. Evidence tagging

- `EvidenceGrade::Measured` is only constructible when the report carries a
  reference device, non-zero paired samples, minimum coverage, and a
  reproducer command string; otherwise the report grades as `Claimed` (real
  data, no reference) or `Synthetic` (generated input). This mirrors
  ADR-291's enforcement-in-types approach and the CLAUDE.md tagging rule.

## Consequences

- RuView can convert vitals claims from CLAIMED to MEASURED with a
  reproducible offline analysis, session by session, scope by scope.
- Honest reporting will likely show heart-rate performance below marketing
  intuition, especially NLOS/moving — that is the purpose.
- CSV ingest means a manual export step per session; acceptable at current
  scale, and the format is the de-facto export of consumer reference gear.
- No clinical claim is implied: agreement statistics against consumer
  reference devices are engineering evidence, not medical validation.

## Validation

- `cargo test -p wifi-densepose-vitals` — CSV rejection cases, alignment
  recovery of known synthetic offsets/drifts, agreement metrics against
  hand-computed fixtures, evidence-grade constructibility rules.
- `cargo bench -p wifi-densepose-vitals` — criterion benchmark for alignment
  over hour-scale synthetic sessions.
- Real-session validation (ESP32 capture + chest strap) remains a follow-up
  requiring hardware evidence per CLAUDE.md.
