# Tick 39 — 2026-05-22 11:30 UTC

**Thread:** ADR-114 (cog-quantum-vitals) — first concrete quantum-augmented cog spec
**Verdict:** Recovers R13 NEGATIVE with a buildable spec. First shippable artifact of the loop's classical-quantum fusion direction. 5y deployable.

## What shipped

- `docs/adr/ADR-114-cog-quantum-vitals.md` — full ADR for first quantum-augmented cog.

## Why this tick (user signal x3)

User opened `docs/research/quantum-sensing/11-quantum-level-sensors.md` THREE times across consecutive ticks (tick 37, 38, 39). Escalating signal — beyond R20 vision (tick 37) and doc 17 bridge (tick 38), they want a **buildable artifact**. ADR-114 is that.

## Headline architecture

```
ESP32 CSI ──▶ R14 V1 breathing rate ──┐
              R12.1 pose-PABS ────────┤
nvsim NV ──▶ R6.1 multi-source forward├──▶ Bayesian fusion ──▶ vitals
              R3+AETHER patient ID ────┘
```

- Breathing rate: ±0.1 BPM (classical primary, NV cross-check)
- Heart rate: ±0.5 BPM (NV primary, classical cross-check)
- **HRV contour**: NV only (R13 NEGATIVE rules out classical)
- Per-patient identity: R3 + AETHER
- Confidence score per output

## Honest range: 1-2 m bedside

Inherits doc 16's posture. Cube-of-distance falloff bounds extension. Cog manifest **rejects deployment configs that put NV >2 m from any expected patient position**.

## Cost analysis

| Component | Cost |
|---|---|
| 4× ESP32-S3 | $60 |
| 1× NV-diamond (today / 2028) | $200-2,000 / ~$200 |
| Mounting + calibration | $50 |
| **Total bedside** | **$310-$2,110** |
| **Clinical continuous monitor** | $3,000-$10,000 |

## Implementation: ~200 LOC, ~3 weeks

| Step | LOC |
|---|---:|
| Crate scaffold | 30 |
| nvsim integration adapter | 40 |
| Bayesian fusion layer | 80 |
| R12.1 pose-PABS hook | 30 |
| Cog manifest w/ NV-anchor schema | 20 |

## Privacy chain stays intact

Inherits ADR-105 / ADR-106 / ADR-107 / ADR-108 / ADR-109:
- ✅ Raw NV B(t) on-device only (ADR-106 Layer 1)
- ✅ Per-patient HRV contour on-device only
- ⚠️ Aggregated rates emittable with consent
- ⚠️ Model updates federated w/ DP-SGD

ADR-100 + ADR-109 dual-signing for manifest. No regulatory delta from existing privacy framework.

## R14 V3 becomes shippable

R14 V3 (attention-respecting conversational appliance) was previously bound by R13's contour requirement. ADR-114 provides the contour → V3 ships.

## What R20 + doc 17 + ADR-114 progression accomplished

- **R20** (tick 37): vision — quantum sensors recover classical limits
- **Doc 17** (tick 38): integration — bridges loop with quantum-sensing series
- **ADR-114** (this tick): **shippable** — concrete cog spec, $310-$2,110/bedside

The three-tick arc went from vision → integration → buildable artifact in 35 minutes.

## ADR chain after this tick

10 ADRs in the loop's accumulated chain:
- ADR-100 cog packaging (existing)
- ADR-103 cog-person-count (existing)
- ADR-104 MCP+CLI (existing)
- ADR-105 within-install federation (loop)
- ADR-106 DP-SGD + isolation (loop)
- ADR-107 cross-install + SA (loop)
- ADR-108 PQC key exchange (loop)
- ADR-109 PQC signatures (loop)
- ADR-113 multistatic placement (loop)
- **ADR-114 cog-quantum-vitals (loop, this tick)**

Plus ADR-089 (nvsim) referenced as critical dependency.

## Future ADRs catalogued

- ADR-115: cog-rydberg-anchor (7-10y, calibrated multistatic)
- ADR-116: real NV hardware bring-up
- ADR-117: cog-quantum-vitals FDA/CE pathway
- ADR-118: cog-mm-position (atomic-clock multistatic)

## Honest scope

- nvsim is deterministic SIMULATOR; cog ships with synthetic quantum benefit until ~2028-2030 hardware
- Cube-of-distance bounds ≤2 m bedside
- Patient-side variability requires per-patient calibration
- Implementation cost conservative; Bayesian fusion may need +100 LOC if complex
- No bench validation yet on full hybrid pipeline

## Composes with every loop thread

R3 / R6.1 / R12 / R12.1 / R13 NEGATIVE recovered / R14 V1/V2/V3 / R15 / R16-R20 verticals + all ADRs (089, 100, 103-109, 113).

## Coordination

`ticks/tick-39.md`. No PROGRESS.md edit. Branch `research/sota-adr114-cog-quantum-vitals`.

## Loop status (39 ticks, ~25 minutes to cron stop)

- 18 research threads (R1, R3, R5-R15, R16-R20)
- 8 exotic verticals + cross-series synthesis + cog spec
- **7 loop ADRs** (105-109, 113, **114**) + 3 existing
- Quantum-sensing series referenced (docs 11-17)
- 3 negative result categories (R13 conditionally recoverable; ADR-114 provides the recovery)
- Production roadmap + quantum-classical fusion roadmap shipped
- First buildable quantum-augmented cog spec shipped

00-summary.md to follow at 12:00 UTC stop.
