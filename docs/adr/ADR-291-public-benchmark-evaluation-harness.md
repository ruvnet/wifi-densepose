# ADR-291: Public-benchmark evaluation harness — Widar3.0 ingest, standard split protocols, leakage guards

- **Status**: Accepted — initial implementation (this PR)
- **Date**: 2026-08-10
- **Deciders**: ruv
- **Tags**: training, evaluation, benchmarks, widar, mm-fi, leakage, honesty

## Context

RuView implements the field's key techniques (CSI ratio, BVP features, MAE
pretraining, rapid adaptation) but reports results only on self-collected data
with self-defined metrics (e.g. the README's held-out temporal-triplet
accuracy). A 2026 deep-research sweep of the WiFi-sensing literature found:

1. Cross-domain generalization is the field's central unsolved problem; the
   only widely reproduced cross-domain result is Widar3.0's BVP benchmark.
2. MM-Fi (NeurIPS 2023) is the standard WiFi-pose benchmark, with defined
   cross-subject and cross-environment protocols.
3. The field had a documented leakage reckoning in 2024–2025: window-level
   random splits on continuous recordings inflate accuracy (one dataset's F1
   collapsed from ~90% to ~22% under subject-disjoint splits — Sensors
   24(10):3159; Signals 6(4):59).

`wifi-densepose-train` already has an `MmFiDataset` NPY loader and a
deterministic `SyntheticCsiDataset`, but no Widar3.0 ingest, no standard split
protocols, and no structural leakage guard. CLAUDE.md already requires
mean-pose baselines and leakage-free held-out splits for pose PCK; nothing in
the code enforces this.

Without leaderboard-comparable numbers, RuView's claims cannot be ranked
against published systems, which blocks both scientific credibility and
commercial (OEM licensing) conversations.

## Options considered

1. **Do nothing; keep self-collected metrics.** Rejected: perpetuates the
   comparability gap.
2. **Port a Python eval stack (SenseFi) alongside the Rust pipeline.**
   Rejected: violates the v2 Rust-workspace direction and adds an unreviewed
   dependency surface.
3. **Extend `wifi-densepose-train` with native loaders + protocol machinery.**
   Chosen.

## Decision

Extend `v2/crates/wifi-densepose-train` with three additions:

### 1. Widar3.0 ingest (`dataset::widar`)

- A parser for the Intel 5300 `.dat` CSI log format ("bfee" records) used by
  the Widar3.0 raw distribution: framed records with a 3-byte header
  (2-byte little-endian length + 1-byte code 0xBB), a 20-byte bfee header
  (timestamp_low, bfee_count, Nrx, Ntx, RSSI a/b/c, noise, agc, antenna_sel,
  len, rate), and a packed 10-bit-per-component complex CSI payload of
  30 subcarrier groups. Invalid records are skipped with a warning, not a
  panic — untrusted file input is validated at the boundary per CLAUDE.md.
- A `WidarDataset` implementing the existing `CsiDataset` trait, mapping
  Widar's `Nrx × Ntx × 30` CSI into windowed `CsiSample`s via the existing
  subcarrier interpolation, with domain metadata (user, room, orientation,
  gesture) parsed from Widar's documented directory/file naming convention.
- No network access: the loader reads a local dataset root. Dataset download
  remains a documented manual step.

### 2. Split protocols (`protocols`)

- A `SplitProtocol` type expressing the standard evaluations: cross-subject
  (MM-Fi style), cross-environment/room, cross-orientation (Widar style), and
  random-baseline (explicitly labelled as leakage-prone, for comparison only).
- Split assignment is a pure function of sample metadata + a seed — fully
  deterministic, no RNG state.

### 3. Leakage guards (`protocols::leakage`)

- A structural `LeakageAudit` that, given a proposed train/test split,
  verifies: (a) subject-disjointness, (b) environment-disjointness where the
  protocol claims it, (c) no two windows from the same continuous recording
  span both sides of the split. A failed audit is an `Err`, not a warning.
- PCK/accuracy reporting requires a `MeanPoseBaseline` computed from the
  training split only, and reports model-vs-baseline together, enforcing the
  CLAUDE.md rule in the type system rather than by convention.
- Evaluation output is an evidence-tagged report (`MEASURED` requires a
  reproducer command line embedded in the report; anything else is emitted as
  `SYNTHETIC` or `CLAIMED`).

## Consequences

- RuView results become comparable to published numbers (Widar3.0 cross-domain
  gesture; MM-Fi cross-subject pose) for the first time.
- The leakage audit will make some existing internal numbers look worse. That
  is the point.
- Parsing a legacy binary format adds maintenance surface; mitigated by
  fixture-based tests with synthetic, deterministically generated `.dat`
  bytes (no dataset redistribution).
- Widar's raw distribution is Intel 5300-specific; ESP32-captured data
  continues through existing loaders. The protocols/leakage machinery is
  loader-agnostic.

## Validation

- `cargo test -p wifi-densepose-train` — unit tests for the bfee parser
  (truncated, corrupt, and valid synthetic fixtures), split determinism,
  leakage-audit rejection cases, and mean-pose baseline math.
- `cargo bench -p wifi-densepose-train` — criterion benchmark for parser
  throughput and split assignment on synthetic corpora.
- No accuracy numbers are claimed by this ADR; it delivers the machinery to
  produce MEASURED ones.
