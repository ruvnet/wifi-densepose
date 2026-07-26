# ADR-275: RF-aware Gaussian spatial memory — the persistent scene representation

| Field | Value |
|-------|-------|
| **Status** | Accepted — **P1 implemented** (`ruview-unified/src/gaussian/`: `primitive.rs`, `map.rs`, `gain.rs`, `graph.rs`; 16 unit tests, criterion benches) |
| **Date** | 2026-07-26 |
| **Parent** | ADR-273 |
| **Relates to** | ADR-030 (persistent field model — superseded in direction by this), ADR-134 (CIR/ISTA), ADR-147 (OccWorld priors), ADR-261 (RuVector graph-ANN — the retrieval layer this memory will index into) |

## 0. PROOF discipline

Grades per ADR-273 §0. The July 2026 external motivators (EmbodiedSplat ~5 fps online semantic Gaussian mapping, ~67× memory efficiency; TGSFormer bounded temporal Gaussian memory; physics-informed channel-gain mapping with incremental Gaussian insertion; JITOMA task-gated activation) are EXTERNAL-UNVERIFIED throughout.

## 1. Context

RuView's spatial state is currently scattered (pose tracker state, field-model eigenstructure, worldgraph tracks). Vision-side SOTA converged on Gaussian fields as the common continuous scene memory, and — the July signal that matters here — the representation crossed into RF: propagation geometry, opacity, attenuation, and scattering as Gaussian primitives, updated *incrementally* when the environment changes. That is exactly the bridge from RuView sensing to a queryable digital twin: one store that answers both geometric questions ("what is near the sofa") and RF questions ("which object caused the channel anomaly", "where did multipath change").

## 2. Decision — the primitive

`RfGaussian` (`primitive.rs`) carries all six ADR-273 attribute groups:

1. **Geometry**: position, per-axis scale (σ), unit-quaternion orientation → anisotropic metric `Σ⁻¹ = R·diag(1/σ²)·Rᵀ`.
2. **Semantics**: 16-d embedding (RuVector-alignable).
3. **RF response**: reflectivity `[4 bands × 4 incident-angle bins]` (2.4/5/6/60 GHz), plus `occupancy` = peak extinction coefficient (nepers/m) used by the gain model.
4. **Motion**: signed Doppler m/s + `{Static, Slow, Fast}` class.
5. **Trust/lifecycle**: confidence ∈ [0,1], timestamp, decay τ, `Provenance {device, model_version, synthetic}`.
6. **Links**: typed references into the scene graph / RuVector entities.

Validated constructor (quaternion normalized, ranges checked); anisotropy and rotation are proven behaviorally (thin axis decays ≥ 80× faster at 0.3 m — the analytic ratio is 86; a 90° quaternion rotates the metric with it).

## 3. Decision — the map

`GaussianMap` (`map.rs`): spatial-hash grid (1 m default pitch) over a flat store.

- **Fusion, not accumulation**: an insert within Mahalanobis² 9 of a same-entity-kind Gaussian merges — confidence-weighted position/scale/occupancy/semantics/reflectivity/Doppler, noisy-OR confidence (`c₁+c₂−c₁c₂`), newest provenance wins, links union. Test: two 0.5-confidence observations 0.1 m apart fuse to one Gaussian at the weighted midpoint with confidence 0.75.
- **Decay + static persistence** (update-loop step 7): exponential confidence decay per Gaussian τ, **stretched by observed lifetime** — `τ_eff = τ·(1 + ln(1 + lifetime/τ))` with `lifetime = last_seen − first_seen` — so a wall confirmed over 30 min outlives a once-seen transient at equal nominal τ (test `long_lived_structure_outlives_transients_at_equal_tau`); prune below 0.02; deterministic (replay test).
- **Merge pass** (update-loop step 5): `merge_overlapping` collapses pairs that are *mutually* inside each other's Mahalanobis gate **and** semantically compatible (cosine ≥ 0.7, or both unlabeled) — orthogonal-semantic overlaps stay separate (test `merge_pass_collapses_mutual_overlaps_but_respects_semantics`). This catches drift the insert-time gate (±1 cell neighborhood only) misses.
- **Queries**: radius (hash + linear reference impl, equivalence-tested on 100-Gaussian grids), kNN (expanding ring), semantic cosine top-k, and the segment-corridor query below.

## 4. Decision — channel gain as a first-class query + inverse update

`gain.rs` implements the RF query surface:

```text
H(tx,rx,f) = (λ/4πd)·e^{-j2πd/λ} · exp(−Σ_g occ_g·I_g)
```

with `I_g` the **closed-form** line integral of each Gaussian's density along the TX→RX segment (1-D Gaussian integral via erf; derivation in the module doc).

**Exactness anchors (MEASURED-CODE):**

- Empty map ⇒ **exact Friis** amplitude (< 1e-15) and propagation phase (`empty_map_returns_exact_friis`).
- Closed-form line integral matches 1 mm trapezoid quadrature through a rotated anisotropic Gaussian to < 1e-6 (`line_integral_matches_numeric_quadrature`).
- On-path absorber attenuates strictly monotonically in occupancy; a 10σ off-path absorber changes LoS gain < 1e-6 dB.

**Inverse update** (`observe_link`) — the incremental-mapping move: measured link amplitude → target optical depth `τ* = ln(friis/measured)`; a projected-gradient step distributes the residual over intersected Gaussians proportional to their path integrals (exact Newton along the link at lr = 1), clamped at occupancy ≥ 0; if nothing intersects and attenuation is demanded, a compact absorber is spawned at the midpoint sized to close the residual. **Measured**: from an empty map, 20 observations of a link with an unseen 0.7-neper (≈6.1 dB) obstruction converge to < 0.06 neper residual and < 0.5 dB prediction error (`inverse_update_learns_a_wall_from_link_residuals`).

## 5. Decision — task-gated scene graph

`graph.rs`: sparse typed nodes (`Object/Room/PersonClass/Device/Event` — person *classes* only; identity lives behind ADR-277's double gate) and relations (`Contains/Near/CausedBy/ObservedBy`). The only sanctioned read is `activate(relevant_kinds, seeds, max_nodes)` — bounded BFS that reports truncation instead of silently scanning (the JITOMA lesson). Tests: an "which object caused the anomaly" activation pulls exactly {event, object, room} and gates out devices/person-classes; the node budget is enforced and truncation is flagged.

## 6. Performance (criterion, release, this machine)

| Benchmark | Result | Note |
|---|---|---|
| `channel_gain`, 1 k Gaussians | **26.9 µs** | was 139 µs with the midpoint-ball candidate query |
| `channel_gain`, 16 k Gaussians | **27.7 µs** | ~O(1) in map size after the corridor rewrite |
| segment corridor query, hash vs linear | 24 µs vs 6 µs (1 k) / 24 µs vs **163 µs** (16 k) | crossover ≈ 4 k Gaussians — reported honestly; both paths kept + equivalence-tested |
| radius query, hash vs linear | 4.3 µs vs 101 µs @ 16 k (23×) | hash loses at 1 k (4.0 vs 1.9 µs) — small maps are brute-force territory |
| `observe_link` inverse update | **74 µs** | was 305 µs pre-optimization |
| map insert+fuse (64 Gaussians, in observe bench setup) | included above | |

The optimization pass replaced a midpoint-ball candidate search (`(2·(L/2+3)+1)³ ≈ 9,300` cell lookups on a 14 m link) with an AABB sweep prefiltered by cell-centre-to-segment distance (bound `margin + √3/2·cell`), after a first corridor attempt (per-sample cube inserts into a BTreeSet) measured *worse* (1.2 ms) and was discarded — kept in this record as the honest negative result.

## 7. Consequences

- The map answers "where is a person likely", "where did multipath change", and "which object caused a channel anomaly" (gain residual → `CausedBy` edge) from one store.
- RuVector integration (ADR-261) becomes: vector search retrieves candidate Gaussians/nodes → graph traversal enforces relations → the gain model *verifies* answers against geometry. The LLM plans the query; it never invents the spatial answer.
- Not yet done (P3): live wiring into `wifi-densepose-sensing-server`, visual/depth Gaussian ingestion, and RuVector index sync.
