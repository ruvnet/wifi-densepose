# ADR-278: Radar inverse rendering + differentiable RF SLAM — a gated research program, not a dependency

| Field | Value |
|-------|-------|
| **Status** | Proposed — research program (deliberately **no code in P1**) |
| **Date** | 2026-07-26 |
| **Parent** | ADR-273 (pillar 6, score 3.6 — highest strategic value, highest hardware + reproduction risk) |
| **Relates to** | ADR-275 (the Gaussian memory these methods would write into), ADR-263/264 (RTL8720F radar platform + wire protocol), ADR-021 (mmWave vitals hardware), ADR-276 (synthetic worlds as the reproduction sandbox) |

## 0. PROOF discipline

Everything numeric in this ADR is **EXTERNAL-UNVERIFIED** — reported by fresh papers/preprints that this repo has not reproduced. That is the point of this ADR: to fix the reproduction gates *before* any of these numbers are allowed to influence the roadmap as if they were ours.

## 1. Context — what the field reports (July 2026)

| System | Claim (theirs) | Availability | Risk read |
|---|---|---|---|
| **RISE** | Single static mmWave radar + multipath inversion → joint room layout + furniture; 16 cm scene Chamfer (baseline 40 cm), 58 % furniture IoU | code available | most reproducible; static sensor matches our appliance posture |
| **DiffRadar** | Radar SLAM + Gaussian fields + differentiable rendering; 0.129 m vs 0.823 m ATE, 94.78 % vs 42.59 % map consistency, 70 fps, 40 MB maps | fresh preprint | treat as **reproduction target, not component** — numbers are single-team, single-venue |
| **GeRaF** | Differentiable RF renderer + SDF + reflectivity, near-range reconstruction; ~32 h on one H100 for 50 k iterations | published setup | offline calibration / digital-twin tool only; unsuitable for continuous adaptation |

The strategic pull is real: all three converge on *inverse rendering into continuous scene representations* — exactly the ADR-275 memory. The risks are equally real: single-source numbers, mmWave hardware variance, and compute profiles (GeRaF) incompatible with edge deployment.

## 2. Decision

1. **No production dependency** on any of these systems or their claims. ADR-275's gain model + inverse update is the only RF-inverse machinery in the deployment path.
2. **Reproduction order: RISE → DiffRadar → GeRaF-lite**, each on one controlled test site, each gated (§3) before the next starts. RISE first because a static radar matches the RuView appliance posture and its inversion writes naturally into `RfGaussian` (occupancy + reflectivity fields already exist for it).
3. **Sandbox-first**: before hardware, each method's core inversion is exercised against ADR-276 synthetic worlds extended with a radar-cube output mode (the `FmcwRadarCube` adapter already normalizes such cubes), so failures separate into "our reimplementation" vs "their claim" cleanly.
4. **Integration contract**: any reproduced system emits into `GaussianMap` via the existing primitive — no parallel scene store. SLAM trajectories, if any, become `Provenance`-stamped map updates subject to ADR-277 export rules like everything else.

## 3. Gates (each phase passes all or the program pauses)

| Gate | Threshold | Split discipline |
|---|---|---|
| G1 RISE-repro (synthetic) | layout Chamfer within 2× of paper's on our synthetic rooms | held-out room geometries |
| G2 RISE-repro (one real site) | qualitative layout recovery + quantified Chamfer vs measured floor plan; report *our* number, whatever it is | site never used in tuning |
| G3 DiffRadar-repro | ATE and map consistency on our trajectory rig; publish the delta vs paper | held-out trajectories |
| G4 Edge viability | inversion or map-update loop ≤ 50 ms p95 on target hardware, or explicit reclassification as offline-calibration tooling (GeRaF's honest category) | — |

A gate failure is a *result*, recorded in this ADR's log — the program exists to convert EXTERNAL-UNVERIFIED into MEASURED, in either direction.

## 4. Consequences

- The roadmap cannot silently absorb preprint numbers; anything radar-inverse must pass through §3.
- ADR-275's primitive already reserves the fields (per-band × angle reflectivity, occupancy, motion) these methods need, so a successful reproduction integrates without schema churn.
- Cost of delay is accepted: pillar 6 scored lowest on readiness, and P1–P4 (encoder, memory, synth, control plane, SRS) do not depend on it.
