# ADR-177: `nvsim` Degenerate-Input Hardening (NV-Diamond Simulator)

| Field | Value |
|-------|-------|
| **Status** | Accepted — 2 real MEDIUM bugs fixed + pinned; determinism preserved |
| **Date** | 2026-06-15 |
| **Deciders** | ruv |
| **Codename** | **NVSIM-FAILCLOSED** |
| **Reviews** | ADR-089 (`nvsim` NV-diamond magnetometer pipeline simulator) |
| **Milestone** | #9 (ungated-crate security sweep) — crate 2 of 4 |

## Context

`nvsim` (ADR-089) is a standalone, **WASM-ready** deterministic NV-diamond
magnetometer pipeline simulator — a forward-only leaf:
`scene → source → propagation → NV ensemble → digitiser → MagFrame + SHA-256
witness`. It has no network surface, so the real attack surface is **degenerate
physical-parameter input** crossing the external boundary — specifically the
WASM `config_json` / `scene_json` entry points.

Two properties matter for this crate that don't for others: it is billed
**deterministic** (a published cross-machine witness must reproduce bit-exactly),
and under `panic=abort` WASM any panic **aborts the whole module**. So a
config-induced panic is a denial-of-service, and a silent numeric corruption
defeats the simulator's entire purpose.

## Decision

Fix the two reachable degenerate-input bugs at their funnel points, each pinned
by a fails-on-old test, **without perturbing the deterministic happy path** (the
guards fire only on non-finite / degenerate input; the published witness is
unchanged).

### Findings fixed (both MEASURED-reproduced)

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| NVSIM-DT-01 | MEDIUM (DoS) | `pipeline.rs:58,95` | `dt = config.dt_s.unwrap_or(1.0 / f_s_hz)`; an external `f_s_hz == 0.0` → `dt = +Inf` → `(dt*1e6) as u64` saturates to `u64::MAX` → `(sample as u64) * dt_us` **panics `attempt to multiply with overflow`** at `sample ≥ 2` (debug/WASM-abort; garbage `t_us` in release). MEASURED: panic at `pipeline.rs:95:30`. | Sanitise `dt` (non-finite/non-positive → 1 µs fallback), cap the `u64` cast at `u64::MAX`, `saturating_mul` the timestamp — no config can overflow it. |
| NVSIM-NAN-01 | MEDIUM (silent corruption) | funnel `digitiser.rs::adc_quantise` (root: near-field clamp bypass in `source.rs`) | A non-finite scene param (NaN/Inf dipole position, Inf moment, NaN loop radius) **bypasses the near-field clamp** (`NaN < R_MIN_M == false` → the `1/r³` path runs → NaN field), and at the ADC `NaN as i32 == 0` (Rust saturating cast) emits a frame `b_pt=[0,0,0]` with **`ADC_SATURATED` CLEAR** — indistinguishable from a legitimate zero-field reading. MEASURED: `b=[NaN,NaN,NaN] sat=false` → `b_pt=[0,0,0] flags=0b0000`. | `adc_quantise`: any non-finite input → code `0` **with the saturation flag raised**; the pipeline's existing `adc_sat` OR-reduction propagates `ADC_SATURATED` onto the frame, making the corruption visible downstream. |

This is the same **NaN-fail-open / NaN-poisoning** family seen across
calibration/vitals/geo and ruview-swarm — non-finite input defeating a guard —
but bounded here to a single frame (no cross-timestep accumulator).

### Dimensions confirmed clean (with evidence)

1. **Determinism integrity — clean.** One RNG only: `ChaCha20Rng::seed_from_u64(seed)`,
   fully caller-seeded (grep: one `seed_from_u64`, **zero** `thread_rng`/`getrandom`/
   `SystemTime`/`Instant`/`HashMap`); `Cargo.toml` pins `rand`/`rand_chacha`
   `default-features=false` (no OS entropy). Box–Muller draws
   `gen_range(f64::EPSILON..=1.0)` (avoids `ln(0)=-Inf` by construction). Frame
   bytes fixed LE; source summation order fixed by `Vec` order. **The published
   cross-machine witness `cc8de9b0…93b4` (`proof_witness_publishes_a_known_value`)
   passes UNCHANGED after both fixes** — the happy path is byte-identical; guards
   touch only degenerate inputs. *Attested caveat (not a finding): libm
   `cos`/`ln`/`sqrt` could differ x86↔wasm; the witness is documented as
   x86_64-captured.*
2. **Panic-free deserialisation — clean.** `MagFrame::from_bytes` validates
   len/magic/version, then per-field `buf[a..b].try_into().expect(...)` are over
   fixed sub-ranges of an already-length-checked 60-byte buffer (provably
   infallible). No `unsafe`, no `panic!`/`unreachable!` in production; every other
   `unwrap`/`expect` is `#[cfg(test)]`.
3. **Div-by-zero / numerical landmines — clean.** `dipole_field`/`current_loop_field`
   clamp `r_norm < R_MIN_M` before `1/r³`,`1/r²` (finite inputs); `shot_noise_floor`
   guards `denom <= 0`; `vec3_normalise` guards `n < 1e-20`. The only hole was the
   NaN *bypass* of the clamp — closed at the ADC funnel (NVSIM-NAN-01).

## Validation

- `cargo test -p nvsim --no-default-features` → **50 → 53** passed, 0 failed (+3 pins:
  `degenerate_zero_sample_rate_does_not_panic`,
  `non_finite_scene_input_flags_frame_instead_of_silently_zeroing`,
  `adc_quantise_flags_non_finite_as_saturated`).
- `cargo test --workspace --no-default-features` → **exit 0**, 0 failed.
- `python archive/v1/data/proof/verify.py` → **VERDICT: PASS**, hash
  `f8e76f21…46f7a` unchanged (nvsim off the signal proof path).
- nvsim's own cross-machine witness `cc8de9b0…93b4` reproduces unchanged.

## Consequences

### Positive
- A config-induced WASM-abort DoS and a silent NaN→fake-zero-field corruption are
  closed at their funnel points, each regression-pinned, with the deterministic
  witness proven intact.

### Negative / Neutral
- None. Guards affect only degenerate inputs; happy-path output is byte-identical.

## Links
- ADR-089 — `nvsim` NV-diamond magnetometer simulator
- ADR-176 — `ruview-swarm` (sibling NaN-fail-open review)
- ADR-172 — core/cli (where the NaN-bug-class root was settled NO)
