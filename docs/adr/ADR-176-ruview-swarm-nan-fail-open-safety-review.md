# ADR-176: `ruview-swarm` NaN-Fail-Open Safety Review

| Field | Value |
|-------|-------|
| **Status** | Accepted — 4 real safety bugs fixed + pinned; 2 issues documented for follow-up |
| **Date** | 2026-06-15 |
| **Deciders** | ruv |
| **Codename** | **SWARM-FAILCLOSED** |
| **Reviews** | ADR-148 (`ruview-swarm` drone swarm control plane) |
| **Milestone** | #9 (ungated-crate security sweep) — crate 1 of 4 |

## Context

`ruview-swarm` (ADR-148) is the drone swarm control plane — hierarchical-mesh
topology, Raft consensus, MARL, CSI sensing payload, MAVLink/PX4 command
dispatch. It is the highest-stakes of the four never-reviewed v2 crates: a defect
here can produce an **unsafe physical drone command**. It had no prior security
ADR.

### Trust-boundary map
Untrusted input enters via `SwarmOrchestrator::receive_peer_state` /
`receive_peer_detection`, which accept full `DroneState` / `CsiDetection` serde
structs with **f64/f32 fields and no finite-check**, and via
`SwarmConfig`/`FhssConfig`/`Geofence` deserialization. The MAVLink wire formats in
`mavlink_messages.rs` are **integer-encoded** (i32 mm / u8) and provably cannot
carry NaN — so the NaN class is reachable through the **serde struct path, not the
MAVLink decode path**. Commands flow out to a `FlightController` (PX4/ArduPilot).

The unifying bug class found: **IEEE-754 NaN/Inf silently defeating a safety
comparison** (`NaN < threshold` evaluates to `false`), causing safety logic to
**fail OPEN**. This is distinct from — but rhymes with — the NaN-state-poisoning
class found earlier in calibration/vitals/geo (there, NaN latched into persistent
state; here, NaN slips through a one-shot guard). Both are "non-finite input
defeats logic," and the fix discipline is the same: **reject non-finite at the
trust boundary, fail CLOSED.**

## Decision

Fix the four reachable fail-open bugs by making each safety predicate
non-finite-aware and fail-closed, each pinned by a fails-on-old test. Document
two further genuine issues that need larger, riskier changes rather than churning
them in a security pass.

### Findings fixed (all MEASURED fails-on-old)

| # | Severity | File:line | Issue | Fix | Pin (old behavior) |
|---|----------|-----------|-------|-----|--------------------|
| F1a | **HIGH** | `failsafe/mod.rs:51` | `nearest_neighbor_dist < collision_dist_m` fails open on a NaN peer position → **collision avoidance silently disabled** | `!is_finite() ||` → `EmergencyDiverge` | `test_nan_neighbor_distance_fails_closed_to_diverge` (old → `Nominal`) |
| F1b | **HIGH** | `failsafe/mod.rs:75` | NaN `battery_pct` bypasses every battery check → drone stays Nominal on unknown battery | `!is_finite() ||` → `ReturnToHome` | `test_nan_battery_fails_closed_to_rth` (old → `Nominal`) |
| F2 | **MEDIUM** | `security/geofence.rs:33` | NaN `z` altitude skips the altitude-breach check and point-in-polygon returns `Safe` → silent geofence bypass | leading non-finite coord → `HardBreach` | `test_nan_altitude_fails_closed` (old → `Safe`) |
| F3 | **MEDIUM/DoS** | `security/antijamming.rs:65,71,102` | empty deserialized `channels_mhz` → `% 0` **panic** in `next_hop`/`current_channel_mhz`/`evasive_hop`/`tick`, crashing the radio task | `len == 0` early-return (`0.0` sentinel) | `test_empty_channels_does_not_panic` (old → panic `divisor of zero`) |
| F4 | **LOW** | `sensing/multiview.rs:70` | NaN `victim_position` passes the `is_some()` filter and propagates into the fused "confirmed victim" location dispatched to the swarm | require finite confidence + position (drop) | `test_nan_victim_position_dropped_from_fusion` (old → non-finite fused position) |

### Dimensions confirmed clean (with evidence)
- **MAVLink decode panic-safety** — `SwarmNodeState::decode(&[u8;20])` `try_into().unwrap()`s are over fixed const ranges of a fixed-size array → provably infallible; no arbitrary-length `&[u8]` decode path exists.
- **UWB/GPS anti-spoofing NaN-safe** — `(gps_dist - uwb_dist).abs() <= tol` already fails CLOSED on a NaN range (counts as inconsistent → spoof rejected); covered by `test_spoofed_gps_invalid`.
- **Bounded grid / no allocate-from-length-field** — `ProbabilityGrid` bounds-checks `cx/cy`; `pos_to_cell` uses saturating `as u32` (no UB).
- **Mesh `nearest_k` NaN-safe sort** — `partial_cmp(..).unwrap_or(Equal)` cannot panic on NaN.
- **No hardcoded secrets** — `MavlinkSigner` key is constructor-injected `[u8;32]`; grep-confirmed nothing embedded.

### Documented, not fixed (genuine — deferred to avoid churn/regression risk)

1. **Raft `AppendEntries` lacks the Log-Matching consistency check**
   (`topology/raft.rs:187`). A follower appends a leader's entries when
   `term >= current_term` **without validating `prev_log_index`/`prev_log_term`**,
   so a malformed/byzantine leader can corrupt a follower's log — a genuine
   consensus-safety gap. A correct fix reworks the log-append plus the
   caller-side vote-tally contract (the existing `handle_message` delegates
   tallying to the caller) — a larger change with test-rewrite risk, so it is
   recorded here rather than rushed in a security pass.
2. **`MavlinkSigner::verify` uses a non-constant-time tag `==` and has no
   replay/timestamp-window rejection** (`security/mavlink_signing.rs:64`). The
   module doc already flags the replay limitation as a demo/test simplification.
   Hardening (constant-time compare + monotonic timestamp window) is a focused
   follow-up.

These two are the recommended scope of the next `ruview-swarm` hardening pass.

## Validation

- `cargo test -p ruview-swarm --no-default-features` → **117 → 123** passed, 0 failed (+6 pins).
- All 6 new tests MEASURED fails-on-old (2× `Nominal`, `Safe`, panic `divisor of zero`, non-finite fused position); pass on the fix.
- `cargo test --workspace --no-default-features` → **exit 0**, 0 failed.
- `python archive/v1/data/proof/verify.py` → **VERDICT: PASS**, hash
  `f8e76f21…46f7a` unchanged (ruview-swarm off the signal proof path).

## Consequences

### Positive
- Four reachable fail-open paths in a *physical-safety* control plane (collision
  avoidance, battery RTH, geofence, anti-jamming radio task) now fail CLOSED on
  hostile/degenerate input, each regression-pinned.
- Extends the "non-finite input defeats logic" defense from the state-poisoning
  variant (calibration/vitals/geo) to the fail-open-comparison variant.

### Negative / Neutral
- Two genuine issues (Raft log-matching, MAVLink signer) remain open by choice —
  see Documented-not-fixed; they define the next hardening pass.

## Links
- ADR-148 — `ruview-swarm` drone swarm control system
- ADR-172 — core/cli review (where the NaN bug-class root question was settled NO)
- ADR-127 — homecore review (sibling NaN/concurrency hardening)
