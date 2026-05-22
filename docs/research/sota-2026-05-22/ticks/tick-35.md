# Tick 35 — 2026-05-22 10:55 UTC

**Thread:** Production roadmap synthesis
**Verdict:** Terminal output of the loop. Maps every research finding to owner / LOC / dependency / priority. Total budget: **~3,500 LOC, ~25 person-weeks**.

## What shipped

- `docs/research/sota-2026-05-22/PRODUCTION-ROADMAP.md` — 6-tier roadmap from loop output to shipped product.

## Headline budget breakdown

| Tier | Timeline | LOC | Person-weeks |
|---|---|---:|---:|
| Tier 1 | Q3 2026 (next quarter) | ~490 | 3-4 |
| Tier 2 | Q3-Q4 2026 | ~1180 | 6-8 |
| Tier 3 | 2027 | ~1140 | 8-10 |
| Tier 4-5 | long horizon | ~700+ | 6-8 |
| **Total** | | **~3,500** | **~25 weeks** |

## Tier 1 (Q3 2026) — 4 items

| # | Item | LOC | Priority |
|---|---|---:|---|
| 1.1 | `wifi-densepose plan-antennas` CLI tool | 360 | HIGH |
| 1.2 | R12.1 pose-PABS in vital_signs cog | 80 | HIGH |
| 1.3 | cog-person-count v0.0.3 chest-centric | 50 | HIGH |
| 1.4 | ADR-029 amendment w/ ADR-113 matrix | 0 | HIGH |

Tier 1 alone delivers: 93× placement-coverage lift, 9.36× intruder-detection lift, ADR-029 closed.

## Tier 2 (Q3-Q4 2026) — 4 items

`ruview-fed` crate (800 LOC), cog-vital-signs DP (120), bench validation (200), MCP placement tool (60).

## Tier 3 (2027) — 4 items

Cross-install fed (530), PQC Phase 1 (490), real-AETHER + R3.2 (200), cog-fall-detection (200).

## Tier 4-5 — long horizon

- 4.x: PQC Phase 2, R10 wildlife cog, R11 maritime cog, R6.1 production
- 5.x: Real RCS measurements, weather-affected propagation, fatigue cog, disaster-fed ethics

## Critical-path graph

```
1.1 CLI ──┬──> 1.3 person-count v0.0.3 ──┬──> 2.1 ruview-fed ──> 2.2 DP-VS ──> 3.1 X-install ──> 3.2 PQC
1.2 R12.1─┘                              │                                   │
                                         └──> 3.3 real-AETHER ──> 3.4 fall  │
                                                                4.x verticals
```

## Why this document matters

After 35 ticks of research output, this is the document that lets a team **pick up and ship** without re-reading the 34 research notes. Priority alignment, estimate-anchoring, critical-path visibility — all in one place.

## What R-numbered threads ship in what tier

| Threads | Tier |
|---|---|
| R5 / R6 / R6.2 family / R6.1 | Tier 1 (placement + PABS) |
| R12 / R12.1 PABS | Tier 1.2 |
| R3 / R3.1 / R3.2 / R14 / R15 | Tier 2-3 (privacy + federation) |
| R7 mincut | Tier 2 (in ruview-fed) |
| R13 NEGATIVE | rules out BP cog, no Tier line |
| R10 wildlife | Tier 4.2 |
| R11 maritime | Tier 4.3 |
| R16/R17/R18 verticals | Tier 4-5 |

## Composes with every loop output

Every loop thread, ADR, vertical sketch has a line in some Tier above. This is the **terminal output** of the loop — the last document that needs the synthesis power of a research loop to produce.

## Honest scope of the roadmap itself

- Estimates are synthetic-data-based; may shift after bench validation
- Critical-path may have hidden dependencies (e.g. AgentDB schema changes)
- 25 person-weeks assumes full-time engineers, not split focus
- Doesn't include integration testing, documentation, deployment ops time
- Tiers are based on architectural dependency, not business priority

## Coordination

`ticks/tick-35.md`. No PROGRESS.md edit. Branch `research/sota-production-roadmap`.

## Loop status approaching completion

~1.1h to cron stop. After 35 ticks the loop has produced:

- 16 research threads (R1, R3, R5-R15, R16, R17, R18)
- 6 exotic verticals (wildlife, maritime, empathic, healthcare, industrial, disaster)
- 6 new ADRs (105, 106, 107, 108, 109, 113)
- 3 negative result categories
- 2 self-corrections
- 3 honest-scope findings
- 9-tick R6 placement family (complete)
- 3-tick R3 cross-room re-ID arc (complete)
- 3-tick R12 structure detection arc (complete)
- This production roadmap synthesis

The 00-summary.md (final tick) will follow after the 12:00 UTC / 08:00 ET cron stop.
