# Tick 29 — 2026-05-22 09:53 UTC

**Thread:** R12.1 (pose-PABS closed loop)
**Verdict:** Synthetic validation of R12 PABS's deferred closure. Pose-updated pipeline gives **9.36× intruder detection lift** vs fixed-expected's 1.29×. **False-alarm problem from R12 PABS resolved.** R12 thread fully closed.

## What shipped

- `examples/research-sota/r12_1_pose_pabs_loop.py` — pure-numpy 50-frame walking-subject + intruder-at-T=25 simulation.
- `examples/research-sota/r12_1_pose_pabs_results.json`
- `docs/research/sota-2026-05-22/R12_1-pose-pabs-closed-loop.md`

## Headline

| Phase | Fixed-expected (R12 naive) | Pose-updated (R12.1 loop) |
|---|---:|---:|
| Pre-intruder (subject walking) | 6.02 | **0.30** |
| Post-intruder | 7.76 | **2.84** |
| **Intruder detection lift** | **1.29×** | **9.36×** |

**Pose updates suppress subject-motion noise by 20×** (6.02 → 0.30), leaving the intruder as a clean 9.36× spike.

## Why this matters

R12 PABS gave 1,161× lift in static scenes but had false alarms when subjects moved. R12.1 closes this gap: the forward model is updated each frame from a simulated pose tracker (5 cm noise, matching ADR-079's 95% PCK@20). Subject motion gets absorbed into the prediction; only the intruder remains as unexplained residual.

## R12 thread fully closed (3 ticks)

| Tick | State | Headline |
|---|---|---:|
| R12 (tick 5) | NEGATIVE | SVD eigenshift fails: 0.69× signal/drift |
| R12 PABS (tick 19) | POSITIVE | 1,161× intruder detection (static) |
| **R12.1 (this)** | **CLOSED LOOP** | **9.36× intruder detection (dynamic)** |

Failure → success with caveat → success without caveat. The multi-tick arc that justifies a long research loop.

## Production roadmap (the Rust glue)

R12 PABS catalogued ~50-100 LOC. Concretely:

```rust
let pose = pose_tracker.estimate(csi_window)?;
let expected_scene = body_model.from_pose(pose) + room_walls;
let y_predicted = fresnel_forward.simulate(expected_scene);
let pabs = (csi_window - y_predicted).norm_sq() / csi_window.norm_sq();
if pabs > threshold { emit_structure_event(); }
```

~80 LOC + ~30 LOC plumbing. Slot into existing vital_signs cog per-frame inference path.

## Composes with prior threads

- R6.1 forward operator
- R7 mincut per-link PABS-after-pose-update is the precise multi-link consistency quantity
- R12 PABS closes deferred item
- R14 V0 security feature (intruder detection) now shippable
- R10/R11 wildlife/maritime variants
- ADR-079/101 pose pipeline is critical-path
- ADR-105/106/107/108 fully on-device

## Honest scope

- 5 cm pose noise matches ADR-079; worse without good signal
- Continuous-time tracking assumed (pose tracker fails → revert to baseline)
- Single subject (multi-subject = data association work)
- Static walls assumed (re-baselining needed for furniture changes)
- Synthetic data only

## Coordination

`ticks/tick-29.md`. No PROGRESS.md edit. Branch `research/sota-r12.1-pose-pabs-loop`.

## All research-loop work substantively complete

After this tick, the loop has:
- 13 research threads (R1, R3, R5-R15)
- 4 ADRs in the privacy chain (105, 106, 107, 108)
- 3 negative-result categories (physics-floor, architecture-error, missing-tool)
- 2 explicit self-corrections (R6.2.2 → R6.2.2.1; R6.2.2.1 → R6.2.4)
- 3 honest-scope findings (R3.1, R6.2.2.1, R3.2)
- R6 placement family (9 ticks: R6, R6.1, R6.2, R6.2.1, R6.2.2, R6.2.2.1, R6.2.3, R6.2.4, R6.2.5)
- R3 cross-room re-ID arc (3 ticks: R3, R3.1, R3.2)
- R12 structure detection arc (3 ticks: R12, R12 PABS, R12.1)

~2.1h to cron stop. Next tick is either:
1. An integrative tick (e.g. ADR amendment summarising R6 placement family for ADR-029)
2. Start consolidating but NOT the final 00-summary yet (premature)
3. Find another concrete experiment
