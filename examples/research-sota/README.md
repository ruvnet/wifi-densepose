# SOTA Research Loop — Examples Overview

Pure-numpy reference implementations from the 2026-05-22 autonomous SOTA research loop. Every script is self-contained, prints headline numbers, and writes a machine-readable JSON result file alongside.

## Folder map

| Folder | Threads | What it covers |
|---|---|---|
| **[01-physics-floor/](01-physics-floor/)** | R1, R6, R6.1 | Bedrock physics — ToA CRLB, single-scatterer Fresnel, multi-scatterer forward model |
| **[02-placement/](02-placement/)** | R6.2 family (7 sub-ticks) | Antenna placement search — 2D / 3D / multi-anchor / chest-centric / multi-subject |
| **[03-spatial-intelligence/](03-spatial-intelligence/)** | R5, R7 | Subcarrier saliency + Stoer-Wagner mincut adversarial defence |
| **[04-rssi/](04-rssi/)** | R8, R9 | RSSI-only counting + RSSI fingerprint K-NN |
| **[05-cross-room-reid/](05-cross-room-reid/)** | R3 arc (3 ticks) | Cross-room person re-identification — naive, physics-informed, embedding-level |
| **[06-structure-detection/](06-structure-detection/)** | R12 arc (3 ticks) | RF-weather eigenshift → PABS → pose-PABS closed loop (NEGATIVE → POSITIVE) |
| **[07-negative-results/](07-negative-results/)** | R13 | Physics-floor scrutiny — why contactless BP from CSI doesn't work |
| **[08-verticals/](08-verticals/)** | R10, R11 | Exotic vertical physics — wildlife (foliage attenuation) and maritime (through-bulkhead) |
| **[09-quantum-fusion/](09-quantum-fusion/)** | R20.1 | Quantum-classical Bayesian fusion (ADR-114 cog-quantum-vitals demo) |

## Running any example

All scripts are pure NumPy. No external dependencies beyond `numpy` itself.

```bash
python examples/research-sota/01-physics-floor/r6_fresnel_zone.py
python examples/research-sota/02-placement/r6_2_5_multi_subject.py
# etc.
```

Each script:
- Prints headline numbers to stdout
- Writes `<script_name>_results.json` next to itself
- Runs in <2 minutes on a laptop (most run in <10 seconds)

## Cross-folder dependency graph

```
01-physics-floor  ──┐
       │            │
       ▼            │
02-placement   ◀────┤
       │            │
       ▼            │
03-spatial-intel ◀──┤
       │            │
       ▼            │
06-structure-detection  ◀──┘
       │
       ▼
09-quantum-fusion  (composes 01+03+06)

04-rssi      (independent, uses 01 forward model)
05-cross-room-reid   (uses 01+03)
07-negative-results  (uses 01)
08-verticals  (uses 01)
```

## Headline findings

| Finding | Source |
|---|---|
| 93× sensing-coverage lift from physics-aware placement | 02-placement (R6.2) |
| 9.36× intruder-detection lift from pose-PABS closed loop | 06-structure-detection (R12.1) |
| 100% coverage of 1-4 occupant household at N=5 anchors | 02-placement (R6.2.5) |
| ~50% breathing-band SNR cost from realistic body multi-scatterer | 01-physics-floor (R6.1) |
| RSSI alone preserves 95% of full-CSI person count | 04-rssi (R8) |
| Stoer-Wagner mincut catches 3/3 adversarial spoofs | 03-spatial-intelligence (R7) |
| Contactless BP/HRV-contour: 5 dB short, physically blocked | 07-negative-results (R13) |
| NV-diamond cardiac magnetometry recovers R13 at bedside | 09-quantum-fusion (R20.1) |

## Related loop output

- **Research notes**: `docs/research/sota-2026-05-22/R{1..20}-*.md`
- **Per-tick summaries**: `docs/research/sota-2026-05-22/ticks/tick-{1..40}.md`
- **Production roadmap**: `docs/research/sota-2026-05-22/PRODUCTION-ROADMAP.md`
- **ADRs from the loop**: `docs/adr/ADR-{105..109,113,114}-*.md`
- **Quantum-sensing series**: `docs/research/quantum-sensing/{11..17}-*.md`

## Honest scope

All scripts are **synthetic-physics derivations**, not bench measurements. Real ESP32-S3 deployments may diverge from these numbers by 5-30% due to multipath, hardware tolerance, environmental drift. Bench validation is the critical next step for any production use; see `PRODUCTION-ROADMAP.md` Tier 2.3.

## Reading order for newcomers

1. Start with **01-physics-floor/R6** (Fresnel forward model) — bedrock for everything else
2. Then **02-placement/R6.2.5** (multi-subject) — practical placement recipe
3. Then **06-structure-detection/R12.1** (pose-PABS) — the security feature
4. Then **09-quantum-fusion/R20.1** (Bayesian fusion) — the future direction
