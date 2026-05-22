# ADR-113: Multistatic anchor placement strategy

**Status:** Proposed · **Date:** 2026-05-22 · **Author:** SOTA research loop tick-31 · **Amends:** ADR-029 (RuvSense multistatic sensing mode)

## Context

ADR-029 (RuvSense multistatic) introduced multi-anchor CSI sensing but did not specify **how many anchors, where to place them, or how zones depend on the target cog**. The SOTA research loop (2026-05-22) produced 9 ticks in the R6 family that quantitatively answer these questions:

- **R6 / R6.1**: Fresnel forward model (single + multi-scatterer)
- **R6.2**: 2D placement search
- **R6.2.1**: 3D placement (ceiling-only fails)
- **R6.2.2**: 2D N-anchor saturation (knee at N=5)
- **R6.2.2.1**: 3D N-anchor (2D knee doesn't hold)
- **R6.2.3**: chest-centric zones (+27 pp gain for vital signs)
- **R6.2.4**: 3D + chest composition (knee at N=6, no ceiling)
- **R6.2.5**: multi-subject union (N=5 hits 100% for 1-4 occupants)

This ADR consolidates the findings into a single placement specification, parameterised by **dimension × zone-mode × occupant-count × cog**.

## Decision

Adopt the **4-axis placement decision matrix** below as the binding RuView installation specification.

### Decision matrix

| Cog category | Dimension | Zone mode | Occupants | Recommended N | Anchor heights | Expected coverage |
|---|---|---|---:|---:|---|---:|
| Presence / occupancy | 2D | body | 1 | 3 | walls @ 0.8 m | 63% |
| Person count | 2D | body | 1-4 | 4 | walls @ 0.8-1.5 m mixed | 86% |
| Pose estimation | 2D | body | 1-2 | **5** | walls @ 0.8/1.5 m mixed | 97% |
| **Vital signs** | 2D | **chest** | 1-4 | **5** | walls @ 0.8/1.5 m | **100%** |
| Pose estimation (3D) | 3D | body | 1-2 | 7-8 | mixed: 0.8/1.5/2.4 m | 65%+ |
| **Vital signs (3D)** | 3D | **chest** | 1-4 | **6** | walls @ 0.8/1.5 m, NO ceiling | **82%** |
| Maritime cabin | 2D | chest | 1-3 | 4 | low (0.5-0.8 m) | 80%+ |
| Wildlife sensing | 1D linear | full-corridor | 1-5 species | 4 (along corridor) | tree-mount mixed | 70%+ |

### Key rules (extracted from R6 family)

1. **Ceiling-only mounting always fails** (R6.2.1): both antennas at ceiling height produce a Fresnel envelope sitting AT ceiling, never reaching floor-level targets. Always include at least one low-anchor.
2. **Vertical link diversity wins in 3D** (R6.2.1): diagonal-in-z links (e.g. 0.8 m → 1.5 m) tilt the ellipsoid through multiple elevations.
3. **Anchor heights should match target zone heights** (R6.2.4): chest-centric zones at z=0.3-1.5 don't benefit from ceiling (z=2.4) anchors. Full-body coverage does.
4. **Chest-centric beats body-centric for vital signs** (R6.2.3): +27 pp coverage gain at N=5 from smaller, occupant-specific zones.
5. **Multi-subject union is the right target for households** (R6.2.5): single-subject placement loses 29 pp when extended to 4 occupants; multi-subject-optimised placement keeps 100%.
6. **N=5 is the consumer recommendation** (R6.2.2 + R6.2.5): the 2D chest-centric multi-subject knee. Beyond N=5, marginal gains are <1 pp.
7. **Avoid placing target zones on the LOS line** (R6.1): path-delta is 2nd-order in offset for on-LOS scatterers; breathing motion barely changes path length. Real installations need subjects OFF the LOS.

### CLI specification (productisation)

The R6.2 CLI tool surfaced through the family ticks:

```
wifi-densepose plan-antennas
    --room W H [Z]                        # 2D or 3D
    --target NAME X Y W H [DX DY DZ]      # repeatable
    --target-mode {body, chest}           # R6.2.3
    --freq-ghz F                          # 2.4, 5.0, 6.0
    --n-anchors N                         # auto-saturate if omitted
    --restarts K                          # 4 default
    --cog COG_NAME                        # auto-select target-mode + N
```

Total LOC for productisation: ~100 LOC on top of the R6.2.5 reference implementation.

### MCP surface (per ADR-104)

```
ruview_placement_recommend(
    room: {width, depth, ceiling?},
    targets: [{name, position, size}],
    cog: str  // auto-configures target-mode + N
) -> {
    anchors: [{x, y, z, height_category}],
    expected_coverage: float,
    placement_rationale: str
}
```

## Alternatives considered

### A. Keep ADR-029 silent on placement

Status: **rejected**. Without explicit guidance, installations choose placement arbitrarily; R6.2 measured **93× spread** between optimal and median placement. Silence is a 93× implicit loss.

### B. Always recommend N=5 + body-centric

Status: **rejected**. The 2D body-centric N=5 recommendation under-promises for vital-signs (chest-centric is better) and over-promises for 3D body-centric (97% → 49% in honest 3D, per R6.2.2.1).

### C. Always recommend N=8

Status: **rejected**. R6.2.2.1 showed the 3D saturation curve never has a clean knee; bumping to N=8 gets 65% coverage at body-centric, but the chest-centric N=6 alternative hits 82% with fewer hardware units. Per-cog decision is the right granularity.

### D. Recommend per-cog without dimension awareness

Status: **rejected**. R6.2.1 + R6.2.2.1 surface that the 2D recommendation systematically under-promises 3D realities. The dimension axis must be explicit.

## Threat model

Placement strategy is not a security-critical decision in itself; coverage gaps create **functional risk**, not adversarial risk. The 4-axis matrix ensures:

| Risk | Mitigation |
|---|---|
| Vital-signs coverage gap | chest-centric + N=5 (or N=6 in 3D) at recommended heights |
| Sleep-monitoring miss | both anchors low (0.5-0.8 m), opposite sides of bed |
| Multi-subject failure | use multi-subject-aware placement (`--target` repeated) |
| Adversarial single-link spoofing | R7 mincut needs N ≥ 4 — placement matrix ensures this for all multi-feature cogs |
| Per-installation variance from documented baseline | CLI tool gives reproducible deterministic placement |

## Consequences

### Positive

1. **Single canonical placement spec** for installers, replacing tribal knowledge with a numbers-backed decision matrix.
2. **Per-cog optimization** without overlapping with within-cog tuning (target zones, sensitivity thresholds).
3. **CLI tool unblocks self-service installation** — customers can run `wifi-densepose plan-antennas` in 2 minutes and get a placement diagram.
4. **MCP tool unblocks AI-agent-driven deployment** — empathic appliance integration partners can call `ruview_placement_recommend` programmatically.
5. **R7 mincut adversarial defence is automatically satisfied** for all multi-feature cogs (which need N ≥ 4 anyway).

### Negative

1. **The matrix is one geometry deep** — 5×5 m bedroom benchmarks. Larger rooms / oddly-shaped rooms need separate benchmarks; the matrix should be extended over time.
2. **Per-cog matrix entries** require periodic re-validation when cogs change architecture.
3. **Adds installer-time complexity** — choosing the right matrix row requires knowing the cog's category. The CLI's `--cog` flag absorbs this.
4. **Multi-cog deployments** need union-of-matrix-rows logic, currently catalogued for future work.
5. **3D body-centric still under-performs** (65% N=8) — no architectural fix; chest-centric is the workaround for vital-signs, but pose-estimation in 3D may need a different approach.

### What this ADR DOES NOT cover

1. **Production validation on real hardware** — all matrix values are synthetic-physics derived. Bench validation on COM5 ESP32-S3 is the next step.
2. **Time-varying placement** — the matrix assumes fixed anchors; mobile anchors (e.g. on a Roomba) are a different regime.
3. **Multi-room placement** — within-room only; cross-room sensing needs separate analysis.
4. **Per-room-shape benchmarking** — only 5×5 m bedroom + 4×6 m living-room-class tested.
5. **Per-frequency matrix variation** — all rows are 2.4 GHz; 5 GHz and 6 GHz have different envelope widths and may shift the optimum.

## Bridge to existing ADRs

- **ADR-029 (RuvSense multistatic)** — **directly amends**: ADR-029's deferred "anchor placement" specification is now this matrix.
- **ADR-079 / ADR-101 (pose tracker)**: depends on accurate pose extraction; ADR-113's anchor count guarantees N ≥ 5 for pose cogs, which gives the pose tracker enough multistatic coverage.
- **ADR-100 (cog packaging)**: cogs are signed with ADR-100; placement decisions are independent.
- **ADR-103 (cog-person-count)**: 2D body-centric N=4 entry maps to this cog.
- **ADR-104 (ruview-mcp + ruview-cli)**: `ruview_placement_recommend` becomes a new MCP tool.
- **ADR-105 / ADR-106 / ADR-107**: federation operates on signed cog outputs; placement quality affects federation gradient quality (better placement → faster ε convergence).
- **ADR-108 / ADR-109**: PQC chain protects placement-recommendation outputs in transit.

## Per-cog target-mode auto-selection

The `--cog` flag in the CLI looks up the cog category and maps to matrix row:

| Cog | Category | Target mode | Heights | N |
|---|---|---|---|---:|
| `cog-presence` | presence | body | low | 3 |
| `cog-person-count` | count | body | mixed low | 4 |
| `cog-pose-estimation` | pose | body | mixed | 5 (2D) / 7 (3D) |
| `cog-vital-signs` | vital signs | **chest** | low+mid | **5 (2D) / 6 (3D)** |
| `cog-breathing` | vital signs | chest | low+mid | 5 (2D) / 6 (3D) |
| `cog-heart-rate` | vital signs | chest | low+mid | 5 (2D) / 6 (3D) |
| `cog-intruder` | structure detection | body | mixed | 5 |
| `cog-maritime-watch` | maritime | chest | low | 4 |
| `cog-wildlife` | wildlife | linear | tree-mount | 4 |

## Connection to research-loop threads

- **R5 (saliency)** — explains why placement maximising Fresnel coverage gives band-spread saliency.
- **R6 / R6.1 (forward model)** — physical foundation.
- **R6.2 family (9 ticks)** — the entire R6.2 family feeds this ADR.
- **R7 (mincut)** — N ≥ 4 satisfied for all multi-feature cogs.
- **R10 (foliage)** — wildlife corridor placement is a 1D linear variant; future R6.2.6 could specialise.
- **R11 (maritime)** — cabin placement is in the matrix.
- **R12 PABS / R12.1** — placement coverage = intrusion-detection sensitivity.
- **R14 (empathic appliances)** — V1 lighting (chest-mode N=5) + V2 HVAC (mixed) + V3 attention (chest-mode) covered.
- **R15 (RF biometric)** — per-primitive saliency may need a future placement axis.

## Honest scope

- **Synthetic physics derivation** — all matrix values come from numpy simulations, not bench measurements. Real-world deployment may shift values by ±5-15%.
- **Single room-geometry baseline** — 5×5 m + 4×6 m. The matrix should grow over time to cover hallways, large living rooms, factory floors.
- **5 cm pose-tracker noise** — assumed in R12.1; degraded pose tracking may invalidate some recommendations.
- **Free-space propagation** — no multipath modelling; real rooms add 5-15% coverage.
- **No furniture occlusion** — sofas, walls, wardrobes ignored.
- **Greedy + 4-restart search** — global optimum may be 1-2 pp higher.

## Implementation plan

| Step | LOC | Owner |
|---|---:|---|
| 1. CLI `--cog` flag with category lookup | 60 | TBD |
| 2. MCP tool `ruview_placement_recommend` | 80 | TBD |
| 3. Per-cog category metadata in cog manifests | 30 | per-cog |
| 4. 3D ellipsoid extension to CLI tool | 50 | TBD |
| 5. Multi-target union to CLI tool | 40 | TBD |
| 6. Integration tests against the R6 family numpy reference | — | TBD |

Total ~260 LOC. Combined with R6.2 productisation (~100 LOC), placement-strategy budget is ~360 LOC.

## Decision-making record

- 2026-05-22 10:06 UTC — drafted by SOTA research loop tick-31 consolidating 9 R6-family ticks. Status: Proposed.
- Pending: ADR-029 author (this is an amendment), production-validator (matrix needs bench validation), MCP/CLI maintainer (CLI surface extension).

## What this ADR closes

The **multistatic placement question** that ADR-029 left open. After this ADR, ADR-029 + ADR-113 + the R6.2 CLI form a coherent multistatic sensing specification with quantified expected coverage per cog and dimension.

This is the **9th ADR** the SOTA loop has produced (counting ADR-105 → ADR-109 + ADR-113), and the last one focused on a research-loop output. Future ADRs (ADR-110/111/112) are operational, not research-driven.

## Closing observation

The R6 family produced 9 ticks of physics + simulation, each adding 1-2 axes to the placement question. ADR-113 collapses all 9 into a single decision matrix that a non-physicist installer can use. **The loop's most ship-relevant integrative output.**
