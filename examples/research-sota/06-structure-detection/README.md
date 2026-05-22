# 06 — Structure detection (R12 arc, 3 ticks)

Detecting "something new in the room" — new furniture, intruder, fallen person. Three-tick arc: R12 NEGATIVE → R12 PABS POSITIVE → R12.1 closed loop.

## Scripts (in arc order)

| Script | Thread | Headline | State |
|---|---|---|---|
| `r12_rf_weather_eigenshift.py` | R12 | Naive SVD-spectrum cosine distance: **0.69× signal/drift = undetectable**. Fails because eigenshift is indistinguishable from natural drift. | **NEGATIVE (missing-tool)** |
| `r12_pabs_implementation.py` | R12 PABS | Physics-Anchored Background Subtraction: **1,161× signal/drift** for unexpected occupant. ~100× lift over R12 NEGATIVE. Achieved by composing R6.1 forward operator as the PABS basis. | **POSITIVE** |
| `r12_1_pose_pabs_loop.py` | R12.1 | Pose-aware closed loop: **9.36× intruder detection in dynamic scenes** (false-alarm problem from R12 PABS resolved). Pose updates suppress subject-motion contribution 20×. | **CLOSED LOOP** |

## The arc summary

R12 (tick 5) → R12 PABS (tick 19) → R12.1 (tick 29): failure → success with caveat → success without caveat.

The arc validates the **research-loop pattern**: catalogue NEGATIVE results explicitly, then revisit them when better tools arrive. R6.1 multi-scatterer (tick 18) provided the tool that R12 was missing in tick 5.

## How PABS works (R12 PABS implementation)

```
y_predicted = sum over voxels of  A(voxel) × reflectivity(voxel)
              where A is the R6.1 forward operator
PABS = ||y_observed − y_predicted||² / ||y_observed||²

If PABS > threshold:
    structural change detected (new scatterer in scene)
```

## How the closed loop works (R12.1)

```
At each frame:
    pose_estimate = pose_tracker.estimate(csi_window)  // ADR-079 / ADR-101
    expected_scene = body_model.from_pose(pose) + walls
    y_predicted = R6.1.forward(expected_scene)
    PABS = ||y_observed − y_predicted||² / ||y_observed||²
    if PABS > threshold:
        emit_structure_event()
```

Subject motion is **absorbed** into the prediction; only unexplained residuals trigger structure events. This is the V0 security feature in R14 + R16/R17/R18 verticals.

## Composition with other folders

- `01-physics-floor/r6_1_multiscatterer.py` provides the A(voxel) forward operator
- `03-spatial-intelligence/r7_multilink_consistency.py` provides per-link adversarial check
- `09-quantum-fusion/r20_1_quantum_classical_fusion.py` composes structure detection with NV-magnetometer fusion

## Production status (per `PRODUCTION-ROADMAP.md`)

- Tier 1.2: R12.1 pose-PABS in `vital_signs` cog (~80 LOC Rust)
- Tier 3.4: Standalone `cog-fall-detection` (~200 LOC)

## See also

- Research notes: `docs/research/sota-2026-05-22/R12-*.md`, `R12_1-*.md`
- ADRs: ADR-079 (pose tracker), ADR-101 (cog-pose-estimation), ADR-029 (multistatic)
