# Tick 31 — 2026-05-22 10:10 UTC

**Thread:** ADR-113 (multistatic placement strategy)
**Verdict:** Consolidates the 9-tick R6 family into a single architectural specification with a 4-axis decision matrix (dimension × zone-mode × occupants × cog). Amends ADR-029. Most ship-relevant integrative output of the loop.

## What shipped

- `docs/adr/ADR-113-multistatic-placement-strategy.md` — full ADR draft.

## The 4-axis decision matrix

| Cog | Dim | Mode | Occ | N | Heights | Coverage |
|---|---|---|---:|---:|---|---:|
| Presence | 2D | body | 1 | 3 | walls 0.8 m | 63% |
| Person count | 2D | body | 1-4 | 4 | walls mixed | 86% |
| Pose | 2D | body | 1-2 | 5 | walls mixed | 97% |
| **Vital signs** | 2D | **chest** | 1-4 | **5** | walls 0.8/1.5 | **100%** |
| Pose | 3D | body | 1-2 | 7-8 | mixed 0.8/1.5/2.4 | 65%+ |
| **Vital signs** | 3D | **chest** | 1-4 | **6** | walls 0.8/1.5 NO ceiling | **82%** |
| Maritime cabin | 2D | chest | 1-3 | 4 | low | 80%+ |
| Wildlife | 1D | linear | 1-5 | 4 | tree mixed | 70%+ |

## Seven binding rules

1. Ceiling-only mounting fails (R6.2.1)
2. Vertical link diversity wins in 3D (R6.2.1)
3. Anchor heights match target zone heights (R6.2.4)
4. Chest-centric beats body for vital signs (R6.2.3)
5. Multi-subject union is the right target (R6.2.5)
6. N=5 is the consumer recommendation (R6.2.2 + R6.2.5)
7. Avoid placing target zones on LOS line (R6.1)

## CLI + MCP productisation surface

```
wifi-densepose plan-antennas
    --room W H [Z] --target ... --target-mode {body,chest}
    --freq-ghz F --n-anchors N --cog NAME
```

```
ruview_placement_recommend(room, targets, cog) -> {anchors, coverage, rationale}
```

~360 LOC total for placement-strategy productisation.

## Per-cog auto-config

| Cog | Mode | N |
|---|---|---:|
| cog-presence | body | 3 |
| cog-person-count | body | 4 |
| cog-pose-estimation | body | 5/7 (2D/3D) |
| **cog-vital-signs** | **chest** | **5/6** |
| cog-breathing | chest | 5/6 |
| cog-heart-rate | chest | 5/6 |
| cog-intruder | body | 5 |
| cog-maritime-watch | chest | 4 |
| cog-wildlife | linear | 4 |

## Why ADR-113 is the loop's most integrative output

The R6 family produced 9 ticks of physics + simulation, each adding 1-2 axes to the placement question. ADR-113 collapses all 9 into a single decision matrix that a non-physicist installer can use.

## Composes with prior threads

- R6.2 family (9 ticks) all feed this ADR
- R7 mincut: N ≥ 4 satisfied for all multi-feature cogs
- R10 / R11: wildlife / maritime entries in the matrix
- R12 PABS / R12.1: placement coverage = intrusion-detection sensitivity
- R14 V1/V2/V3: all matrix rows covered
- ADR-029: directly amended

## Honest scope

- Synthetic physics derivation; bench validation pending
- Single room-geometry baseline (5×5 m bedroom + 4×6 m living-room class)
- 5 cm pose-tracker noise assumed (R12.1)
- Free-space, no multipath, no furniture occlusion
- Greedy + 4-restart search

## ADR chain after this tick (9 loop ADRs)

| # | ADR | Status |
|---|---|---|
| 1 | ADR-105 | within-install fed |
| 2 | ADR-106 | DP + isolation |
| 3 | ADR-107 | cross-install + SA |
| 4 | ADR-108 | PQC key exchange |
| 5 | ADR-109 | PQC signatures |
| 6 | **ADR-113** | **multistatic placement** |

Plus 3 already shipped before the loop (100, 103, 104). 9 ADRs total in the privacy + federation + provenance + placement chain.

## Coordination

`ticks/tick-31.md`. No PROGRESS.md edit. Branch `research/sota-adr113-multistatic-placement`.

## Loop's research + architecture output substantively complete

After 31 ticks, the loop has produced everything addressable in the cron-driven 8-min unit:
- 13 research threads (R1, R3, R5-R15)
- 6 ADRs (105-109, 113) closing privacy + federation + provenance + placement
- 3 negative-result categories (physics-floor, architecture-error, missing-tool-revisited)
- 2 explicit self-corrections
- 3 honest-scope findings
- 9-tick R6 placement family
- 3-tick R3 cross-room re-ID arc
- 3-tick R12 structure detection arc (NEGATIVE → POSITIVE → CLOSED LOOP)

~1.8h to cron stop. Remaining time can be used for:
1. Continue with new ADRs (ADR-110/111/112 catalogued but operational, not research-driven)
2. Cross-thread integration experiments
3. Eventually write the 00-summary.md after 12:00 UTC stop
