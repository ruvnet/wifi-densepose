# ADR-172: `wifi-densepose-cli` + `wifi-densepose-core` CSI-Deserialiser Security Review

| Field | Value |
|-------|-------|
| **Status** | Accepted — clean-with-evidence, 4 regression pins added |
| **Date** | 2026-06-15 |
| **Deciders** | ruv |
| **Codename** | **CSI-DESERIALISER-HARDENING** |
| **Supersedes / amends** | none (records review; references ADR-127 §9 for the `core` portion, ADR-136 for the pre-existing DoS ACs) |

## Context

The beyond-SOTA security sweep (branch `feat/v2-beyond-sota-sweep`) reviewed each
`v2/` crate for real, reproducible defects. Two crates had no prior dedicated
security ADR:

- **`wifi-densepose-core`** — the dependency root for all 12 downstream crates
  (types, traits, error types, CSI frame primitives). A defect here is a
  force-multiplier: every consumer inherits it.
- **`wifi-densepose-cli`** — the user-facing entrypoint
  (`calibrate`/`calibrate-serve`/`enroll`/`train-room`/`room-watch` + MAT-gated),
  which parses untrusted UDP CSI packets and operator-supplied paths.

A **specific hypothesis** motivated the core review. Three earlier reviews in
this campaign found a systemic **NaN-state-poisoning bug class** in crates that
depend on core (`wifi-densepose-calibration`, `-vitals`, `-geo`): a non-finite
(NaN/Inf) input latched into persistent filter/accumulator state (IIR `y1/y2`,
running mean, Welford/von-Mises accumulator, voxel grid) → silent **permanent**
feature failure. The load-bearing question for this review: **does that bug class
originate in a shared `wifi-densepose-core` primitive** (making the right fix a
single root fix), or was it independently re-implemented in each downstream
crate (making the three existing local fixes complete)?

## Decision

Record the review outcome and lock in the existing DoS guards with regression
tests. **No production code is changed** — both crates were already hardened
(ADR-136 acceptance criteria + `sanitize_room_id`); the gap was *untested*
guards, which a future refactor could silently remove.

### Load-bearing question — VERDICT: **NO** (the NaN class does not live in core)

`wifi-densepose-core` exposes **no stateful accumulator of any kind** — no
Welford/running-mean, no von-Mises/circular-mean, no IIR/biquad filter state, no
voxel grid.

- **MEASURED:** `grep` over `core/src` for
  `welford|von_mises|biquad|y1|y2|running_mean|accumulat|voxel|self.*+=` matched
  only the `InvalidState` *error* enum variant, "reset state" doc comments, and a
  test-only LCG — **zero** stateful logic. The only float math in core is
  construction-time projection (`CsiFrame::new` → amplitude/phase via `mapv`) and
  pure stateless `utils` functions; nothing persists across frames.
- **Corroboration:** `wifi-densepose-calibration::Features::from_series`
  (`extract.rs:103–133`) already filters non-finite samples → `Features::ZERO`.
  The downstream fixes are independently re-implemented, confirming each crate
  rolls its own accumulator and each local fix is correct and complete. **A fix
  in core would be a no-op (there is nothing to fix).**

Consequence: the NaN-state-poisoning class is a *downstream-local* pattern, not a
core-rooted defect. No hidden fourth instance exists in the shared primitive.

### Findings (all pins — guards already present, now tested)

| # | Location | Guard (pre-existing) | Regression pin | Evidence (MEASURED) |
|---|----------|----------------------|----------------|---------------------|
| 1 | `core` `types.rs:801` `from_canonical_bytes` | `saturating_mul` shape-vs-length check before `Vec::with_capacity(rows*cols)` | `canonical_decode_oversized_shape_is_bounded_not_allocated` | With guard removed: **panics `capacity overflow` at `types.rs:801`**; with guard: passes |
| 2 | `core` `types.rs` decoder | typed `CanonicalDecodeError`, never panics | `canonical_decode_never_panics_on_arbitrary_bytes` (fuzz sweep) | panic-free on arbitrary bytes |
| 3 | `cli` `calibrate.rs:276–291` | length check `buf.len() < 20 + n_pairs*2` before `Array2::zeros(n_antennas*n_subcarriers)` | `test_parse_csi_packet_oversized_claim_is_rejected_not_allocated` | 255×65535 claim in a 2 KB packet → `None` (no allocation) |
| 4 | `cli` `calibrate.rs` parser | `None`-returning on malformed input | `test_parse_csi_packet_never_panics_on_arbitrary_bytes` (fuzz sweep) | panic-free on arbitrary UDP bytes |

### Dimensions confirmed clean (with evidence)

1. **Panic-on-adversarial-input = 0** — `from_canonical_bytes` returns a typed
   error for every malformed class; `parse_csi_packet` returns `None`. Both
   fuzz-swept panic-free.
2. **NaN handling** — `Confidence::new` rejects NaN
   (`!(0.0..=1.0).contains(&NaN)` ⇒ `Err`); `compute_bounding_box` /
   `to_flat_array` are NaN-tolerant (f32 min/max ignore NaN).
3. **Empty-frame safety** — `amplitude_variance` / `mean_amplitude` are
   panic-free on an empty `Array2` (ndarray 0.17 returns finite / `None`).
4. **Unbounded-memory DoS** — bounded in both deserialisers (findings 1 & 3).
5. **Path traversal** — `calibrate-serve` defends every client-supplied
   `room_id`/`bank`/`baseline` via `sanitize_room_id` (`[A-Za-z0-9_-]`, 64-char
   cap) with existing tests; bearer-auth gate + non-loopback-bind warning present.
   `mat export` writes to an operator-supplied `PathBuf` (acceptable CLI behavior).
6. **Secrets** — `--token` is read from `CALIBRATE_TOKEN` env, never embedded.

## Validation

- `cargo test -p wifi-densepose-core` → **35 → 37** lib passed, 0 failed (+3 doctests)
- `cargo test -p wifi-densepose-cli --no-default-features` → **24 → 26** passed, 0 failed
- `cargo test --workspace --no-default-features` → **exit 0**, 0 failed
- `python archive/v1/data/proof/verify.py` → **VERDICT: PASS**, hash
  `f8e76f21a0f9852b70b6d9dd5318239f6b20cbcb4cdd995863263cecdc446f7a` **unchanged**
  (core/cli are off the signal proof path — confirms no pipeline alteration)

## Consequences

### Positive
- Two CSI deserialisers (the untrusted-input boundary of both the library root
  and the network-facing CLI) now have their DoS guards pinned against
  regression — a future refactor that drops a length check fails CI.
- The NaN-state-poisoning class is settled as downstream-local; reviewers no
  longer need to suspect a shared-root defect, and the three prior local fixes
  are confirmed complete.

### Negative
- None. Test-only change; no behavior or API change.

### Neutral
- The `core` portion is also noted in ADR-127 §9 (shared security-review log);
  this ADR is the canonical record for the `wifi-densepose-cli` review.

## Links
- ADR-127 — HOMECORE state machine (shared security-review log, §9)
- ADR-136 — pre-existing CSI deserialiser DoS acceptance criteria
- ADR-151 — per-room calibration (`calibrate`/`calibrate-serve` surfaces)
