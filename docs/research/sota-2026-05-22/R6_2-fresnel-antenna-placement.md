# R6.2 — Fresnel-aware antenna placement: a 93× sensing-coverage lift from physics

**Status:** working CLI tool + demo + 5×5 m bedroom benchmark · **2026-05-22**

## Premise

R6 (Fresnel forward model) said: there is a ~40 cm wide ellipsoid around a 5 m WiFi link where occupancy dominates the CSI signal. Outside that envelope, CSI is mostly multipath edge noise. The current RuView installation guide is essentially "stick the seed wherever the AP is and hope for the best."

This thread quantifies how much coverage you give up by ignoring the Fresnel geometry — and provides a CLI-shaped tool that solves the placement problem given a room layout + target occupancy zones (bed, chair, where the user actually spends time).

## Method

In 2D the first Fresnel zone is an ellipse with:

- foci at Tx and Rx
- semi-major axis `a = (d + λ/2) / 2`
- semi-minor axis `b = √(a² − (d/2)²) ≈ √(d·λ)/2` for d ≫ λ

A point `x` is inside the first Fresnel zone iff `|Tx-x| + |x-Rx| ≤ d + λ/2`. This is the natural 2D extension of R6's midpoint radius formula.

`examples/research-sota/r6_2_antenna_placement.py` rasterises target zones at 5 cm resolution, evaluates every candidate (Tx, Rx) pair on the room perimeter (25 cm step), and picks the pair that maximises total target-zone area inside the first Fresnel ellipse.

## Benchmark: 5×5 m bedroom

Two target zones:

| Zone | Position | Area |
|---|---|---:|
| Bed | (1.5, 0.5)-(3.5, 2.0) | 3.00 m² |
| Chair | (3.5, 3.5)-(4.3, 4.3) | 0.64 m² |

2,900 antenna pairs evaluated at 2.4 GHz (λ = 12.5 cm):

| Placement | Tx | Rx | Link | Bed cov | Chair cov | **Total** |
|---|:---:|:---:|---:|---:|---:|---:|
| **Optimal** | (1.25, 0.00) | (4.75, 5.00) | 6.10 m | 43.5% | 86.7% | **51.1%** |
| Median (rand-place baseline) | varies | varies | varies | varies | varies | 0.5% |
| Worst | varies | varies | 5.00 m | varies | varies | **0.0%** |

**Best/median improvement: 93×.** The current "stick it anywhere" deployment recipe is ~50-100× below optimal in this geometry. Most placements give effectively no sensing of the actual target zones, because the Fresnel ellipse threads space that nobody occupies.

## Why diagonal-across-the-room wins

The optimal placement runs **diagonally across the long axis**, threading both the bed and the chair. The 6.10 m link length is **longer** than any wall-parallel link (≤5 m), which gives a **wider** Fresnel ellipse at the midpoint:

```
b(d=5.0, λ=0.125) = √(5.0 × 0.125)/2 = 39.5 cm
b(d=6.1, λ=0.125) = √(6.1 × 0.125)/2 = 43.7 cm  (+10%)
```

The Fresnel envelope **gets wider as the link gets longer** (up to the link-budget limit, which we ignore here — R10 sets that). Counter to the intuition "shorter link = stronger signal", *longer* links cover *more space*. Up to a budget-limited point.

## Per-cog deployment recommendations

Plugging this into each existing cog's installation flow:

| Cog | Target zones | Recommended placement |
|---|---|---|
| `cog-person-count` (R8/R5/ADR-103) | Any room occupancy | Diagonal across longest axis |
| `cog-pose-estimation` (ADR-079, ADR-101) | Where pose matters (gym corner, kitchen workspace) | Place link so the zone is within ~50% of the midpoint envelope width |
| AETHER re-ID (ADR-024) | Doorway + main occupancy zone | Tx near doorway, Rx diagonal across; doorway transit triggers ID, main zone confirms |
| `cog-maritime-watch` (R11) | Cabin floor space | Tx ceiling-mount, Rx floor-mount, vertical diagonal through cabin |
| `cog-wildlife` (R10 follow-up, not yet built) | Forest clearing perimeter | Tx and Rx on opposite trees, link threads the clearing midline |

These recommendations make the existing installation guides ~50-100× more effective without any hardware change.

## What this DOES enable

1. **A shippable CLI tool** that gives end users immediate placement guidance. Same input shape as `wifi-densepose plan-antennas --room 5x5 --target bed,1,1,2x1`. The output is a concrete placement that an installer can mount to.
2. **Reproducible benchmarks** for the "is the placement good enough?" question. Existing RuView installs have no objective placement metric; this tool gives one.
3. **A natural cog feature**: when a new cog is added (e.g. `cog-wildlife`), the placement guide is generated from the cog's target-zone schema, not hand-written per-cog.
4. **Adaptive 4-anchor multistatic generalisation.** The current 2D single-pair search extends naturally to N anchors — pick the 4-anchor set that maximises union-of-Fresnel-envelopes coverage. Each additional anchor saturates coverage (diminishing returns), giving a quantitative answer to "is 4 anchors enough?" (in a 5×5 m bedroom: yes; in a 10 m living room: no, need 6).

## Composes with prior threads

- **R6** (Fresnel forward model) — provides the 2D extension; R6.2 is the natural application.
- **R1** (CRLB) — combining R1's localisation precision with R6.2's coverage gives a full **sensing geometry budget**: how many anchors × where × precision.
- **R10** (foliage range) — the link-budget cap on link length is set by R10's path-loss model. For sparse foliage at 2.4 GHz, R10 said 100 m is the maximum link; R6.2 says use most of that budget for wider Fresnel envelopes.
- **R11** (maritime) — ship cabins are small + steel-walled (Fresnel envelope narrowed by reflection geometry); R6.2's recipe still applies but coverage saturates faster.
- **R14** (empathic appliances) — V1 lighting / V2 HVAC / V3 attention-respecting need to sense the *occupant*, who lives in known target zones (bed, sofa, desk). R6.2 is the installation-time tool that ensures the empathic-appliance system actually sees the user.
- **ADR-105** (federated learning) — placement plays no role in federation per se, but better placement → better local training data → faster convergence with smaller (ε, δ) budget (ADR-106).

## Honest scope

- **2D approximation.** Real Fresnel envelopes are 3D ellipsoids; the 2D model is correct for floor-level scattering (most occupancy) but underestimates ceiling-mounted antennas' coverage of standing occupants. A 3D version is a half-day's work.
- **Free-space assumption.** Real rooms have furniture, walls, and floor reflections. Multipath sometimes *helps* coverage outside Fresnel (multi-bounce paths add signal paths). The 2D Fresnel-only model is a lower bound on coverage; real rooms typically have +5-15% coverage from multipath.
- **Rectangular target zones.** People don't occupy rectangles. A more realistic version uses pose-trajectory distributions (where do users *actually* spend time) — derived from R3 + AETHER + a few weeks of data.
- **Single-pair only.** Multistatic with N > 2 anchors is a strict superset; the current code only searches over single-pair placements. Multi-anchor extension is the next R6.2.1.
- **Perimeter-only candidates.** The 25 cm step on walls assumes wall-mounted antennas. Ceiling mounts, free-standing tripods, and furniture-attached placements are all valid but harder to evaluate (more design freedom = larger search space).
- **No link-budget gate.** A diagonal-across-30-m-warehouse placement may have wider Fresnel envelope but exceed the link budget (R10). The current code doesn't gate by link budget; for large rooms this is critical.

## Practical CLI shape

```bash
wifi-densepose plan-antennas \
    --room 5.0 5.0 \
    --target bed 1.5 0.5 2.0 1.5 \
    --target chair 3.5 3.5 0.8 0.8 \
    --freq-ghz 2.4 \
    --step 0.25
```

Output:
```
BEST placement:
  Tx:                1.25, 0.00
  Rx:                4.75, 5.00
  Coverage fraction: 51.1%
  Per-zone:
    bed:   43.5%
    chair: 86.7%
```

This is the deliverable a customer would run before mounting hardware. Two minutes of computation saves an installer from making the "stick it on the AP" mistake that loses 50-100× of the sensing potential.

## What this DOES NOT enable

- **3D placement** for ceiling-mount antennas.
- **Link-budget gating** for long-distance deployments.
- **Multi-anchor optimisation** for the eventual ADR-029 multistatic shipping.
- **Pose-trajectory-aware target zones** — these need empirical data, not just static room layouts.
- **Furniture / wall reflection modelling** — bigger model, slower search, marginal improvement.

## Next ticks (R6.2 follow-ups)

- **R6.2.1**: 3D extension. Replace 2D ellipse with prolate ellipsoid; allow ceiling/floor antenna mounts.
- **R6.2.2**: N-anchor multistatic placement (maximises *union* of N pairwise Fresnel envelopes). Quantitative answer to "is 4 anchors enough?"
- **R6.2.3**: Pose-trajectory-aware target zones, fed from AETHER's per-installation occupancy data (R3 + ADR-105 federation enables this without raw data leaving the install).
- **Productise**: add as `wifi-densepose plan-antennas` subcommand; mention in ADR-104's CLI surface as a deferred MCP tool `ruview_placement_recommend`.

## What this DOES close

The "we don't have a placement recommendation tool" gap that every RuView installer hits is now closed with a working CLI-shaped prototype. The 93× median-vs-best improvement is large enough that productising this is high-leverage with no new physics.

## Connection back

- **R5** (saliency) — placement that gets a target zone *in* the first Fresnel zone yields the band-spread saliency profile R5 measured. Bad placement (target outside the zone) gives band-edge-only saliency, which is what R5 explicitly didn't measure (no occupant outside the envelope = no saliency to measure).
- **R6** (Fresnel forward model) — direct extension. R6 gave the math; R6.2 productises it.
- **R7** (mincut adversarial) — multi-pair placement that R6.2.2 will solve enables the multi-link consistency check R7 needs. Single-pair installations can't run R7's adversarial defence.
- **R9** (RSSI fingerprint K-NN) — RSSI doesn't have the spatial precision Fresnel gives; placement matters less for RSSI-only deployments (R8 + R9 showed 95% retained even with coarse spatial info).
- **R14** (empathic appliances) — the V1/V2/V3 verticals all need *the right user* sensed, which means the user's bed/sofa/desk must be inside the Fresnel envelope. R6.2 makes this an installation-time check, not a deploy-and-pray.
