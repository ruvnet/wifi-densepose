# 04 — RSSI-only sensing

RSSI is the simplest CSI summary (one number per packet). These scripts quantify what's recoverable from RSSI alone vs full CSI.

## Scripts

| Script | Thread | Headline |
|---|---|---|
| `r8_rssi_only_count.py` | R8 | RSSI-only person count: **59.1% accuracy = 94.82% of full-CSI v0.0.2** with a tiny 656-parameter MLP. RSSI keeps 95% of counting capacity. |
| `r9_rssi_fingerprint_knn.py` | R9 | Cosine-NN on RSSI fingerprints: **2.18× lift** over chance (MODERATE). Surfaces counting-vs-localization asymmetry: RSSI is great for count, weaker for per-location ID. |

## The counting-vs-localization asymmetry

R8 + R9 together demonstrate that RSSI:
- **Retains 95% of person-count capacity** (R8)
- **Retains only ~30% of localization capacity** (R9)

This means RSSI-only deployments (the cheap path) are viable for **occupancy / count** but inadequate for **per-occupant features** (vitals, identity, pose).

## When to use RSSI-only

Per ADR-113 placement matrix, RSSI-only is appropriate for:
- `cog-presence` (binary occupancy)
- `cog-person-count` (occupant count)
- Very cost-sensitive deployments (chicken-scale R19 livestock, for instance)

NOT appropriate for:
- `cog-vital-signs` (needs CSI per-subcarrier shape)
- `cog-pose-estimation` (needs CSI multistatic geometry)
- `cog-quantum-vitals` (ADR-114, needs CSI fusion with NV)

## See also

- Research notes: `docs/research/sota-2026-05-22/R8-*.md`, `R9-*.md`
- Composes with: `01-physics-floor/` (uses Fresnel forward model insight)
