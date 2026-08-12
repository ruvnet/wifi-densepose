# ADR-317: Multi-domain benchmark scorecard — regressions cannot hide behind pooled accuracy

- **Status**: Accepted — initial implementation planned (ADR-300 phase 1)
- **Date**: 2026-08-11
- **Deciders**: ruv
- **Tags**: benchmark, aetherarena, ci-gate, evidence, honesty, domain-generalization, substrate

## Context

This ADR is primitive 17 of the perception-substrate program (ADR-300) and the
per-PR enforcement edge of the phase-1 certificate spine. In the ADR-300
dependency DAG it reads accuracy from the evidence engine (ADR-304), consumes
the domain state produced by out-of-distribution detection (ADR-302), scores
against calibration certificates (ADR-301), and is anchored in the witness chain
(ADR-319). It is the surface that makes the rest of the spine testable on every
change to sensing code.

A single pooled accuracy number is the classic way a domain-generalization
regression hides. A model can raise mean PCK or mean presence accuracy while
quietly collapsing on unseen rooms, unseen devices, or stationary subjects —
exactly the conditions WiFi sensing fails in and exactly the conditions a
pooled average washes out. The strategic assessment (ADR-300) named this: what
distinguishes infrastructure from a demo is that a regression on *any* operating
domain is caught before merge, not discovered in the field.

RuView does not need a new benchmark to do this. AetherArena is already
**v0-complete infrastructure** (ADR-149): a deterministic scoring engine
reusing `wifi-densepose-train` (`src/ruview_metrics.rs`, `src/ablation.rs`,
`src/eval.rs`, `src/proof.rs`), a `PROOF_SEED=42` determinism substrate that
SHA-256-hashes outputs against an expected hash, an append-only witness ledger,
and a live Hugging Face Space. ADR-145's ablation harness already computes
presence accuracy, localization error, FP/FN, latency percentiles, a
privacy-leakage score, and **cross-room degradation**. The board is
intentionally empty (benchmark-first). What is missing is not a scorer but a
**scorecard format** that reports per-domain rather than pooled, and a
**sensing-crate CI gate** that runs it on every PR.

## Options considered

1. **Keep the single pooled score / `RuViewTier`.** Rejected: it is exactly the
   surface a per-domain regression hides behind; a Gold tier can coexist with a
   broken unseen-room slice.
2. **Add a new benchmark repo/harness for domains.** Rejected: AetherArena's
   scorer, determinism binding, and witness ledger already exist and are the
   right engine; a parallel harness would fork the scoring substrate and its
   anti-gaming/leakage discipline.
3. **Extend the AetherArena scorer with a per-domain scorecard and wire it as a
   per-PR sensing-crate gate.** Chosen.

## Decision

Reuse the AetherArena scorer and witness ledger (ADR-149) and add two things: a
**multi-domain scorecard** format and a **sensing-crate PR gate** that produces
it.

### 1. The multi-domain scorecard

The scorecard reports each capability broken out by operating domain, never
pooled into one figure. The v0 domain axes:

- **Presence**: `room-known`, `room-unseen`, `device-unseen`, `stationary-10m`
  (a stationary subject at range — the canonical WiFi failure case).
- **Pose**: `matched`, `subject-unseen`, `room-unseen`.
- **OOD rejection**: the rate at which genuinely out-of-distribution input is
  correctly returned as UNKNOWN by ADR-302 (a capability, not a failure) and
  the false-UNKNOWN rate on in-distribution input.
- **Calibration drift**: fingerprint-distance trajectory against the ADR-301
  certificate over the scored window, and the fraction of inferences in each
  ADR-302 `DomainState` (KNOWN / DEGRADED / UNKNOWN).

Each cell carries exactly one `EvidenceLevel` (L0–L5, ADR-282). A slice scored
on synthetic input is L0/`Synthetic` by construction; a slice on a leakage-free
held-out real split is graded higher and only then may a per-domain number be
labelled MEASURED. Pose PCK cells additionally require the mean-pose baseline
and a leakage-free held-out split (CLAUDE.md) or they are not reported as pose
accuracy at all.

### 2. Per-domain regression gate

- The gate compares each scorecard cell against the merged-baseline scorecard
  stored in the AetherArena witness ledger. A regression **in any single
  domain** beyond its configured threshold fails the PR, even if the pooled
  average improved. Improvement on `room-known` cannot buy a regression on
  `room-unseen`.
- Thresholds are per-domain and per-capability; the unseen/stationary/OOD
  domains carry the strictest budgets because they are the ones a pooled score
  hides. The baseline is append-only and witness-anchored — a new baseline is a
  new signed ledger entry, never an in-place overwrite (ADR-149 ledger pattern,
  ADR-319 anchoring).

### 3. Sensing-crate CI wiring

- Every PR that touches a sensing crate runs the scorecard across all domains
  under the ADR-011/ADR-149 determinism binding (`PROOF_SEED=42`), so the run
  is reproducible and tamper-evident. The gate is added to
  `.github/workflows/` as an authoritative check.
- The held-out real split remains private and is never accessible to synthetic
  generation, augmentation, or calibration (ADR-149 leakage constraint, ADR-282
  rule d). Submitters/PRs provide a model, not predictions on data they hold.

### Provenance and honesty discipline

- No benchmark numbers are invented by this ADR. It delivers the scorecard
  format, the per-domain gate, and the CI wiring; the numbers come from the
  ADR-304 evidence ledger and the AetherArena scorer on real data, labelled at
  the honest evidence level. Empty domains report "no evidence," which the gate
  treats as no coverage — never as a pass.

## Consequences

- A domain-generalization regression can no longer merge behind a flattering
  pooled average; the failure mode that most distinguishes fielded sensing from
  a demo is caught at PR time.
- Every PR touching sensing pays a per-domain scoring cost. Bounded by reusing
  the existing deterministic scorer and by tiered compute (CPU smoke vs full
  score, ADR-149), but it is a deliberate cost for per-domain safety.
- The empty AetherArena board fills with honest, per-domain, evidence-labelled
  results rather than a single headline tier — consistent with the
  benchmark-first posture and with ADR-282's ecosystem positioning.
- Some domains will show weak or absent coverage. Surfacing that per-domain is
  the point; the scorecard must never paper over a thin domain with a pooled
  number.
- The program-level acceptance test (ADR-300) is encoded here as an AetherArena
  scenario, closing the loop once the phase-1 spine lands.

## Validation

- `cargo test` on the AetherArena scorer extension — per-domain slicing math
  against fixtures; per-domain regression gate fails on a single-domain
  regression while pooled improves, and passes when all domains hold; empty
  domains report "no evidence," not a pass; every cell carries exactly one
  `EvidenceLevel`; synthetic slices are L0 by construction.
- Determinism: a scored run reproduces its SHA-256 hash under `PROOF_SEED=42`
  (ADR-011/ADR-149 binding); the baseline scorecard is append-only and
  witness-anchored (ADR-319), never mutated in place.
- CI: the sensing-crate gate runs on a PR touching a sensing crate and blocks a
  planted single-domain regression.
- Real-data scorecards (a leakage-free held-out split with ADR-303 references)
  are the maturity milestone; a synthetic scorecard is L0 and no per-domain
  number is MEASURED without a reproducer per CLAUDE.md.
