# ADR-276: Physics-guided synthetic RF world generator — randomize physics, not textures

| Field | Value |
|-------|-------|
| **Status** | Accepted — **P1 implemented** (`ruview-unified/src/synth/`: `room.rs`, `raytrace.rs`, `generator.rs`; 10 unit tests + the ADR-273 acceptance pipeline consumes it end-to-end) |
| **Date** | 2026-07-26 |
| **Parent** | ADR-273 |
| **Relates to** | ADR-015 (MM-Fi/Wi-Pose datasets), ADR-089 (nvsim — the determinism pattern this follows), ADR-135 (empty-room baselines the generator can emulate) |

## 0. PROOF discipline

Grades per ADR-273 §0. WaveVerse (released simulator, phase-coherent ray tracing) and HybridSim (92.07 % vs 54.22 % synthetic-only→real activity recognition when physics is modeled explicitly) are EXTERNAL-UNVERIFIED motivators. Every output of this generator is stamped `RfModality::Synthetic` and every number derived from it is labeled SYNTHETIC — that stamp survives into `Provenance.synthetic` at the ADR-277 export boundary.

## 1. Context

RuView's scarcest resource is labeled, *diverse* RF data: rooms, materials, antenna placements, people, chipsets. The 2026 evidence says synthetic RF transfers **when the physics is explicit and the randomization hits physical parameters** (permittivity, geometry, kinematics, hardware nuisances) rather than cosmetic noise. A physics generator also gives the ADR-273 acceptance machinery something it can never get from captures alone: *ground truth by construction* and unlimited strict-split diversity.

## 2. Decision — physics core

### 2.1 Rooms and materials (`room.rs`)

Shoebox rooms `[0,Lx]×[0,Ly]×[0,Lz]`, one wall material with **complex permittivity** `ε = ε_r − j·σ/(ωε₀)` and normal-incidence Fresnel reflection `Γ = (1−√ε)/(1+√ε)`. Presets (concrete/drywall/glass, ITU-R P.2040 ballpark) plus a perfect absorber for test isolation. Measured sanity: concrete at 2.4 GHz gives |Γ| ≈ 0.39–0.45 with phase inversion; |Γ| < 1 for all passive presets; ε_r = 1, σ = 0 gives Γ = 0 exactly. People are validated-in-room point scatterers with constant velocity and RCS.

### 2.2 Multipath (`raytrace.rs`)

Allen–Berkley image method, reflection order ≤ 2 (per-axis images `±x + 2nL`, bounce count `|2n|` / `|2n−1|`), plus single-bounce bistatic person scattering with amplitude `√(σ_rcs/4π)/(d₁·d₂)` (bistatic radar equation, amplitude form):

```text
H(f) = Σ_paths Γ^order · (c/f)/(4π) · s_p · e^{−j2πf·d_p/c}
```

**Doppler is never injected** — it emerges from the person's path length changing between snapshots.

**Physics gates (MEASURED-CODE):**

| Gate | Test | Result |
|---|---|---|
| Direct path ≡ Friis | `direct_path_is_exact_friis` | < 1e-15 per subcarrier (absorber walls) |
| Reciprocity `H(a→b) = H(b→a)` | `channel_is_reciprocal` | < 1e-12, with person + concrete walls |
| Image geometry | `first_order_reflection_matches_mirror_geometry` | floor/ceiling bounce at exactly the mirror distance; 1 direct + 6 first-order + second-order set |
| Doppler | `moving_person_produces_the_analytic_doppler_phase_rate` | residual-phase rotation matches `−2πf·Δd/c` to < 1e-6 rad across 4 steps |

## 3. Decision — domain randomization (`generator.rs`)

Per room, seeded ChaCha20 (nvsim discipline — same seed ⇒ byte-identical corpus, cross-machine):

- **Physics**: dimensions 4–10 × 3–8 × 2.4–3.2 m; ε_r ∈ [2,7], σ ∈ [0.002,0.1] S/m; random TX/RX placements; person start/heading/speed/RCS.
- **Hardware nuisances** (what breaks naive models in the field): per-room gain ×0.5–2, static phase offset, **CFO drift** ±0.3 rad/snapshot, thermal noise, 5 % packet loss (snapshot re-delivery), 3 % wideband interference bursts.
- **Provenance for strict splits**: every window carries a full `PartitionKey` (room/day/person/chipset/firmware/layout) so ADR-273 §4 holdouts exist by construction.

Measured: byte-determinism per seed (and divergence across seeds); presence windows carry > 5× the temporal amplitude variance of empty windows (actual measured ratio on the test corpus is far higher); labels/keys complete.

The CFO nuisance earned its keep immediately: it *defeated the first tokenizer* (empty rooms looked like motion) and forced the CFO-alignment step now documented in ADR-274 §3.1 — exactly the class of failure a physics-parameter randomizer exists to surface before real deployments do.

## 4. What this generator is NOT

- Not a WaveVerse replacement: order-2 specular + point scatterers, no diffraction, no diffuse scattering, no angle-dependent Fresnel, no antenna patterns. These are refinements to add *when a P2 real-data gap analysis demands them*, not before.
- Not evidence of real-world accuracy: the ADR-273 acceptance numbers on this data validate the pipeline; the synthetic→real transfer claim (HybridSim-style) is untested here and stays EXTERNAL-UNVERIFIED until P2 replay experiments.

## 5. Performance

Criterion (release): 1 room × 4 windows × 3 links generates in **3.1 ms** (≈ 260 µs/window) — corpus generation is never the bottleneck; the 8-room acceptance corpus builds in well under a second even in debug.

## 6. Consequences

- Every pipeline stage gains a deterministic, physics-proven test bed; regressions in adapters/tokenizer/encoder now fail loudly against ground truth.
- Data scarcity stops gating architecture work: strict-split experiments (rooms/chipsets/layouts) run in CI.
- The honest-labeling chain (`RfModality::Synthetic` → `Provenance.synthetic` → SYNTHETIC-graded ADR claims) is structural, not editorial.
