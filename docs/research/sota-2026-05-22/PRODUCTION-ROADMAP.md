# Production roadmap: from loop output to shipped product

**Status:** synthesis — every loop finding mapped to a concrete next-step action · **2026-05-22**

## Why this document exists

The SOTA research loop produced 34+ ticks of physics, simulation, architecture, and vertical sketches. Without a roadmap, none of it ships. This document maps every loop output to:

- **Owner** (which team / role picks it up)
- **LOC estimate** (rough engineering cost)
- **Dependencies** (what must land first)
- **Priority** (HIGH/MEDIUM/LOW based on leverage × certainty)

Reading order: top sections are the highest-leverage / shortest-path-to-ship items. Bottom sections are exotic / long-horizon work.

## Tier 1 — Ship in next quarter (Q3 2026)

### 1.1 — `wifi-densepose plan-antennas` CLI tool

**Source ticks**: R6.2 / R6.2.1 / R6.2.2 / R6.2.2.1 / R6.2.3 / R6.2.4 / R6.2.5 / ADR-113
**Owner**: CLI maintainer (per ADR-104)
**LOC**: ~360 (placement search engine, 4-axis matrix lookup, 3D ellipsoid extension, multi-target union)
**Dependencies**: none (reference numpy implementations exist in examples/research-sota/)
**Priority**: **HIGH** — 93× sensing-coverage lift from physics alone; existing customers can re-mount today

```bash
wifi-densepose plan-antennas \
    --room 5 5 [Z] \
    --target NAME X Y W H [DX DY DZ] \
    --target-mode {body, chest} \
    --cog COG_NAME \
    --freq-ghz 2.4 \
    --n-anchors N
```

### 1.2 — R12.1 pose-PABS closed loop in `vital_signs` cog

**Source ticks**: R12 PABS / R12.1 / R6.1
**Owner**: `vital_signs.rs` maintainer
**LOC**: ~80 (PABS = ||observed − predicted||² / ||observed||², coupled with pose_tracker.rs updates)
**Dependencies**: existing pose pipeline (ADR-079, ADR-101), R6.1 multi-scatterer forward operator
**Priority**: **HIGH** — 9.36× intruder-detection lift; ships a V0 security feature

### 1.3 — `cog-person-count` v0.0.3 with chest-centric placement

**Source ticks**: R5 / R8 / R6.2.3 / ADR-113
**Owner**: cog-person-count maintainer (ADR-103)
**LOC**: ~50 (placement-aware training config + per-cog `--target-mode=body` default in ADR-113 matrix)
**Dependencies**: 1.1 CLI tool
**Priority**: **HIGH** — already shipped v0.0.2 from this loop's K-fold + label-smoothing work; v0.0.3 is the placement-aware retrain

### 1.4 — ADR-029 amendment with ADR-113 placement matrix

**Source**: ADR-113
**Owner**: ADR-029 author / architect
**LOC**: 0 (ADR amendment only)
**Dependencies**: 1.1 CLI tool (validates the matrix)
**Priority**: **HIGH** — closes the multistatic-placement question ADR-029 left open

## Tier 2 — Ship in next 6 months (Q3-Q4 2026)

### 2.1 — `ruview-fed` crate (within-installation federation)

**Source**: ADR-105 + ADR-106
**Owner**: federation specialist (new role)
**LOC**: ~800 (Krum aggregator, LoRA+int8 delta codec, MERIDIAN centroid hook, mincut consistency check, DP-SGD with Moments Accountant, primitive isolation enforcement)
**Dependencies**: AgentDB, ruvllm-microlora, ruvector-mincut (all existing)
**Priority**: **HIGH** — enables R14 empathic appliances + R16/R17/R18 vertical work; ~3-week effort

### 2.2 — Updated `cog-vital-signs` with R15 primitive isolation

**Source**: R14 / R15 / ADR-106
**Owner**: vital-signs cog maintainer
**LOC**: ~120 (PrimitiveTag enum, on-device-only enforcement at API surface, per-cog config schema)
**Dependencies**: 2.1 `ruview-fed`
**Priority**: **HIGH** — privacy-compliant medical-grade vitals; required for R16 healthcare deployment

### 2.3 — Bench validation suite for placement matrix

**Source**: ADR-113 honest scope
**Owner**: bench engineer + COM5 hardware
**LOC**: ~200 (test fixtures + CSI capture + matrix-vs-observed comparison)
**Dependencies**: 1.1 CLI tool
**Priority**: **MEDIUM** — turns ADR-113's synthetic numbers into validated numbers

### 2.4 — MCP tool `ruview_placement_recommend`

**Source**: ADR-104 + ADR-113
**Owner**: ruview-mcp maintainer
**LOC**: ~60
**Dependencies**: 1.1 CLI tool
**Priority**: **MEDIUM** — enables AI-agent-driven deployment

## Tier 3 — Ship in next year (2027)

### 3.1 — Cross-installation federation (ADR-107)

**Source**: ADR-107
**Owner**: federation + crypto specialist
**LOC**: +530 (Bonawitz secure aggregation, threshold Shamir, PKI client, per-installation rotation key)
**Dependencies**: 2.1 `ruview-fed`
**Priority**: **MEDIUM** — enables R16-R17-R18 cross-installation cogs

### 3.2 — PQC migration Phase 1 (ADR-108 + ADR-109)

**Source**: ADR-108 + ADR-109
**Owner**: crypto specialist
**LOC**: +220 (Kyber-768 KEM) + +270 (Dilithium-3 signing) = +490 total
**Dependencies**: 3.1 cross-installation federation
**Priority**: **MEDIUM** — opt-in pgc-hybrid mode; required by Phase 2 (2027-Q2)

### 3.3 — Real-AETHER + R3.2 embedding-level cross-room re-ID

**Source**: R3 / R3.1 / R3.2 / ADR-024
**Owner**: ML training engineer
**LOC**: ~200 (R3.2 protocol composed with ADR-024 contrastive head)
**Dependencies**: ADR-024 AETHER training (~1-2 days on RTX 5080)
**Priority**: **MEDIUM** — produces working cross-room re-ID, unblocks R14 per-occupant features

### 3.4 — `cog-fall-detection` (R12.1 production)

**Source**: R12.1 + ADR-079
**Owner**: cog developer
**LOC**: ~200 (pose-PABS pipeline + fall-event detector + EHR/alert integration shim)
**Dependencies**: 1.2 R12.1 in vital_signs
**Priority**: **HIGH** for R16 healthcare; **MEDIUM** for general

## Tier 4 — Long horizon (2027-2030)

### 4.1 — PQC migration Phase 2 (hybrid default)

**Source**: ADR-108 + ADR-109 Phase 2
**Owner**: crypto specialist
**LOC**: +150
**Dependencies**: 3.2 Phase 1 deployed and stable
**Priority**: **MEDIUM** — CNSA 2.0 compliance

### 4.2 — Wildlife cog (R10 + cog-wildlife)

**Source**: R10
**Owner**: ecology partner + cog developer
**LOC**: ~300 (gait-frequency classifier + species-prior model + labelled wildlife CSI dataset)
**Dependencies**: 2.1 federation (for cross-deployment training), labelled dataset (external partnership)
**Priority**: **LOW** — high impact but long lead-time for data

### 4.3 — Maritime cog (R11 + cog-maritime-watch)

**Source**: R11
**Owner**: maritime partner + cog developer
**LOC**: ~250 (through-seam acoustic-coupled CSI + man-overboard detector + crew-vitals)
**Dependencies**: 2.1 federation, maritime partner for ship deployment
**Priority**: **LOW** — niche but high-value-per-deployment

### 4.4 — R6.1 multi-scatterer in production `vital_signs`

**Source**: R6.1
**Owner**: vital-signs maintainer
**LOC**: ~150 (replace scalar Fresnel with multi-scatterer forward; PPE-aware variant for R17 industrial)
**Dependencies**: 1.2 R12.1 first
**Priority**: **MEDIUM** — improves SNR-budget accuracy; PPE variant for R17

## Tier 5 — Research-needed (post-2027)

### 5.1 — R6.1 with real body RCS measurements

**Source**: R6.1 honest scope
**Owner**: physics consultant + bench engineer
**LOC**: 0 (paper, measurement campaign)
**Dependencies**: anechoic-chamber access
**Priority**: **LOW** — refines per-body-part reflectivity by 2-3×

### 5.2 — Outdoor / weather-affected propagation

**Source**: R10 / R11 / R17 / R18 honest scope
**Owner**: physics consultant
**LOC**: 0 (paper)
**Dependencies**: weather-station data
**Priority**: **LOW** — needed for outdoor cogs

### 5.3 — Long-shift gait fatigue (cog-worker-fatigue)

**Source**: R17 + R10
**Owner**: ergonomics + ML developer
**LOC**: ~300 (temporal gait-drift detector)
**Dependencies**: labelled multi-hour worker data
**Priority**: **LOW** — OSHA-aligned but long lead-time

### 5.4 — Disaster-deployment federation with consent

**Source**: R18
**Owner**: ethics consultant + legal
**LOC**: 0 (policy work)
**Dependencies**: FEMA / urban-SAR partnerships
**Priority**: **LOW** — ethical work first, technical later

## Tier 6 — Operational / management

### 6.1 — Owner-key rotation policy (ADR-111)

**Source**: ADR-109 honest scope
**Owner**: security architect
**Priority**: **MEDIUM** — required before ADR-109 Phase 1

### 6.2 — Cross-organisation PKI bootstrapping (ADR-107 operational)

**Source**: ADR-107 deferred items
**Owner**: ops architect
**Priority**: **MEDIUM** — needed before cross-installation federation goes multi-org

### 6.3 — FDA / CE regulatory pathway (R16)

**Source**: R16 healthcare honest scope
**Owner**: regulatory consultant
**Cost**: $500K-$2M per device class
**Timeline**: 6-18 months
**Priority**: **HIGH** for healthcare deployment

## Critical-path graph (text version)

```
1.1 plan-antennas CLI ----+
                          v
1.2 R12.1 vital_signs ---+
                         v
1.3 cog-person-count v0.0.3 ---+
                               v
2.1 ruview-fed crate --------+
                             v
2.2 cog-vital-signs DP -----+
                            v
3.1 cross-install fed -----+
                           v
3.2 PQC migration --------+
                          v
3.3 R3.2 embedding cross-room
3.4 cog-fall-detection (independent of 3.3)
4.x verticals (R10, R11, R16, R17, R18)
```

## Total engineering budget across the loop's output

| Tier | LOC | Person-weeks |
|---|---:|---:|
| Tier 1 (Q3 2026) | ~490 | 3-4 |
| Tier 2 (Q3-Q4 2026) | ~1180 | 6-8 |
| Tier 3 (2027) | ~1140 | 8-10 |
| Tier 4-5 (long horizon) | ~700+ | 6-8 |
| **Total** | **~3,500 LOC** | **~25 person-weeks** |

This includes both the privacy + federation + PQC chain (~1,820 LOC) and the placement / cog / integration work (~1,700 LOC).

## What this roadmap DOES enable

1. **A team can pick this up and start shipping** without re-reading the 34 research notes.
2. **Priority alignment** for engineering managers.
3. **Estimate-anchoring** for project planning.
4. **Critical-path visibility** for parallel work scheduling.

## What this roadmap DOES NOT enable

- Production validation (still required per Tier 2.3 bench validation).
- Regulatory approval (Tier 6.3 separate pathway).
- Partnership establishment (Tier 4.4 / 4.3 / 5.4 all need external partners).
- The roadmap is **only as good as the underlying ticks** — synthetic-data-based estimates may shift.

## Composes with every loop thread

This document is the **terminal output** of the loop — every research thread, ADR, vertical sketch, and follow-up has a line in some Tier above.

## Connection back

Every loop output → roadmap line:
- Research threads R1, R3, R5–R18 → Tier 3-5 cogs + Tier 1-2 implementations
- ADRs 105-109 + 113 → Tier 2-4 implementation work
- R6 family (9 ticks) → Tier 1.1 CLI + Tier 4.4 production multi-scatterer
- R3 arc (3 ticks) → Tier 3.3 real-AETHER + Tier 3 cross-room re-ID
- R12 arc (3 ticks) → Tier 1.2 R12.1 pose-PABS + Tier 3.4 cog-fall-detection
- Negative results (R12 revisited, R13 floor, R3.1 architecture) → Tier 5 research-needed items
- Honest-scope findings → Tier 5 research-needed items
