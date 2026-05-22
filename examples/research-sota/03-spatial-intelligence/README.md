# 03 — Spatial intelligence

Subcarrier-selection and multi-link adversarial-defence primitives.

## Scripts

| Script | Thread | Headline |
|---|---|---|
| `r5_subcarrier_saliency.py` | R5 | Top-8 most-informative subcarriers for person-count classification. Band-spread (not band-clustered) — explained by R6 Fresnel forward model (zone-1 occupancy has flat per-subcarrier phase). Max/mean ratio 2.85×. |
| `r7_multilink_consistency.py` | R7 | Stoer-Wagner minimum cut detects **3/3 adversarial spoofs** across multi-link CSI graphs. Identifies which links were compromised; reports "physically impossible CSI" via topological consistency. |

## Why these compose

- **R5 saliency** tells us WHICH subcarriers carry the discriminative information for each cog.
- **R7 mincut** tells us WHEN we should trust the CSI at all (multi-link consistency check).

Both feed into the production cogs (R8 RSSI counter uses R5's top-8 subcarriers; R12 PABS uses R7's per-link consistency check as an adversarial defence layer).

## Sample output

```
=== R5 subcarrier saliency ===
Top 8 most informative subcarriers (out of 56):
  [41, 52, 30, 31, 10, 35, 2, 38]
  Max/mean ratio: 2.85x (band-spread, not band-clustered)

=== R7 mincut adversarial detection ===
Scenario A (no adversary):    detected as compromised: []
Scenario B (link 0 spoofed):  detected as compromised: [0]   ✓
Scenario C (link 2 spoofed):  detected as compromised: [2]   ✓
Scenario D (multi compromised): detected as compromised: [1,3] ✓
Detection rate: 3/3
```

## See also

- R5 explained by `01-physics-floor/r6_fresnel_zone.py` (band-spread = zone-1 occupancy signature)
- R7 used by `06-structure-detection/r12_pabs_implementation.py` and `09-quantum-fusion/`
- Research notes: `docs/research/sota-2026-05-22/R5-*.md`, `R7-*.md`
