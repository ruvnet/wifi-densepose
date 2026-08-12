# ADR-294: WiFi Veil integration — emission-shaping countermeasure as an advisory BFLD dependency

- **Status**: Accepted — initial implementation (this PR)
- **Date**: 2026-08-10
- **Deciders**: ruv
- **Tags**: privacy, bfld, bfi, wifi-veil, countermeasure, dependency

## Context

RuView's BFLD layer (ADR-118, ADR-141) senses via beamforming feedback while
enforcing structural privacy invariants on data entering the node. The 2026
research sweep identified the complementary, unaddressed surface: a node's own
*outgoing* BFI is unencrypted and enables passive third-party
re-identification (BFId, ACM CCS 2025); IEEE 802.11bf-2025 shipped with no
privacy mechanism; and no commercial product occupies the countermeasure
category.

[`wifi-veil`](https://github.com/ruvnet/wifi-veil) (codename VEIL, extracted
from this monorepo as a standalone crate) models a compliant emission-shaping
defense: keyed Givens rotations over the fine subspace of compressed
beamforming reports, energy-preserving (never jamming), reversible by a
keyed legitimate receiver. The crate is dependency-free, deterministic,
std-only, WASM-ready, dual MIT/Apache-2.0, and explicitly SYNTHETIC/L0: it
models waveform controls and never drives a radio.

RuView should consume this capability rather than re-implement it, giving the
sensing stack a defensive counterpart under one evidence regime.

## Options considered

1. **Vendor the veil sources into a RuView crate.** Rejected: forks the
   witness-pinned upstream and duplicates maintenance.
2. **crates.io dependency.** Not yet available (v0.1.0 unpublished at
   decision time); revisit when released.
3. **Git dependency pinned to an exact rev, feature-gated in
   `wifi-densepose-bfld`.** Chosen.

## Decision

- Add `wifi-veil` to `v2/Cargo.toml` `[workspace.dependencies]` as a git
  dependency pinned to rev `018468b5d2bf41f35c552910f35659830af0eb91`
  (v0.1.0). Exact-rev pinning preserves provenance and reproducibility for a
  pre-release upstream; bumping the rev is an explicit, reviewable change.
- Gate it in `wifi-densepose-bfld` behind a new `veil` feature
  (`veil = ["std", "dep:wifi-veil"]`), off by default — the default build
  remains dependency-light and unchanged.
- New `bfld::veil` module (advisory-only):
  - `ShieldAssessment`: stable projection of wifi-veil's deterministic
    attacker-vs-protector `ExperimentReport` (re-ID accuracy shield-off/on,
    chance level, throughput ratio, energy-conservation audit), always
    carrying the `SYNTHETIC/L0` evidence label.
  - `assess` / `assess_default`: run the deterministic experiment.
  - `optimized_shield`: wrap `hyper_optimize` to derive the
    optimizer-shipped shield config plus its verifying assessment.
- Boundaries, stated structurally and in docs:
  - **Advisory only.** Nothing in the integration emits RF, alters frames,
    or relaxes any BFLD gate/invariant (I1–I3 untouched).
  - **Evidence honesty.** Every veil-derived figure is labeled
    `SYNTHETIC/L0`; no MEASURED claim is possible from this path (hardware
    validation lives in wifi-veil's own P5 roadmap).
  - ESP32 nodes cannot shield their own feedback (per wifi-veil's platform
    matrix); the integration therefore informs posture and reporting, not
    on-node emission control.

## Consequences

- RuView gains a sense-and-defend posture no commercial offering has, under
  a single claim taxonomy.
- First git dependency in the workspace: builds now fetch one pinned
  external rev. Acceptable: the crate is dependency-free, small, witness-
  pinned upstream, and license-compatible (MIT OR Apache-2.0 into MIT).
- Feature-gated consumers (e.g. sensing-server privacy reporting, the
  desktop UI) can surface shield assessments later without new deps.
- When wifi-veil publishes to crates.io, switch the workspace entry to a
  version requirement in a follow-up ADR amendment.

## Validation

- `cargo test -p wifi-densepose-bfld --features veil` — determinism,
  shield-reduces-re-ID, compliance (energy conservation), chance-band
  attainment, evidence labeling, optimizer wrapper.
- `cargo test -p wifi-densepose-bfld` (default features) — unchanged
  behavior with the feature off.
