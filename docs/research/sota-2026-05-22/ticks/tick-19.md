# Tick 19 — 2026-05-22 07:44 UTC

**Thread:** R12 PABS implementation
**Verdict:** **R12 NEGATIVE → POSITIVE.** PABS detects unexpected occupants at **1,161× natural drift floor** vs R12 naive SVD's 11× — a **~100× lift** purely from using physics-grounded prediction.

## What shipped

- `examples/research-sota/r12_pabs_implementation.py` — pure-numpy PABS over R6.1's multi-scatterer forward operator.
- `examples/research-sota/r12_pabs_results.json` — full 6-scenario benchmark.
- `docs/research/sota-2026-05-22/R12-pabs-implementation.md` — research note documenting the NEGATIVE → POSITIVE conversion.

## Headline benchmark

| Scenario | PABS / drift | SVD (R12 baseline) / drift |
|---|---:|---:|
| Empty room (subject missing) | **7,362×** | 65× |
| Subject as expected (sanity check) | 0× | 0× |
| +1 new furniture | **84×** | 11× |
| +1 unexpected human | **1,161×** | 11× |
| Subject moved 10 cm | 21,966× | 90× |
| Natural drift floor (5% wall) | 1× | 1× |

## Why this is the meta-positive result

Two negative results in this loop (R12, R13). R12 has now been **revisited and turned positive** by using a tool (R6.1's multi-scatterer forward operator) that didn't exist when R12 was first run. This is the meta-lesson:

> A research loop that catalogues NEGATIVE results creates a backlog of revisitable work that pays off when later tools become available. R12 → R12 PABS is a worked example.

R13 cannot be similarly revisited — its 5 dB shortfall is a hard physics floor, not a missing model.

## The subject-moved-10cm caveat

Scenario F gives PABS=22,000×, which looks like a bug but is correct behaviour. PABS detects **any** structural mismatch between expected and observed. Real production PABS needs a **pose-aware forward model** that updates the expected scene from `pose_tracker.rs` in real-time. The actual structure-detection signal is **PABS-after-pose-update**.

This is ~50-100 LOC of Rust glue. Catalogued as R12.1 follow-up.

## Composes with everything

- **R6.1** unblocked this implementation
- **R7** gets precise per-link consistency definition (residual norm small on all links → no structure; spike on one → either local structure OR compromised link; mincut disambiguates)
- **R11** (maritime) enables container-tamper / hatch-seal applications
- **R12 NEGATIVE** → POSITIVE
- **R14** (V0 security feature) intruder detection without biometric storage
- **ADR-029** needs to reference PABS as the structure-detection primitive
- **R10** (foliage) PABS-vs-forest works if canopy modelled or learned

## Honest scope

- Pose-PABS closed loop not yet built (every subject move = false alarm)
- Synthetic data only; real-world drift floor needs measurement
- Population-prior body; per-subject body would tighten residual
- Single time-frame (real pipeline needs temporal averaging)

## Coordination

`ticks/tick-19.md`. No PROGRESS.md edit. Branch `research/sota-r12-pabs-implementation`.

## Remaining work

- **R12.1**: pose-PABS closed loop
- **R12.2**: localised residual decomposition (where is the structural change)
- **R12.3**: real-world validation on bench ESP32 captures
- **R3 follow-up**: physics-informed env_sig prediction
- **R6.2.1**: 3D ceiling/floor placement
- **R6.2.3**: chest-centric / pose-trajectory zones
- **ADR-107**: cross-installation federation w/ secure aggregation

~4.3h to cron stop. **19 ticks landed. 1 NEGATIVE result revisited and turned POSITIVE.**
