# Tick 8 — 2026-05-22 05:25 UTC

**Thread:** R6 (Fresnel forward model)
**Verdict:** Working closed-form forward model + numpy demo. Bedrock physics that the entire `wifi-densepose-signal` DSP pipeline implicitly assumes is now explicit.

## What shipped

- `examples/research-sota/r6_fresnel_zone.py` — pure-numpy Fresnel-zone radius + per-subcarrier phase prediction. Four canonical scenarios over 802.11n/ac 20 MHz channels (52 subcarriers, 312.5 kHz spacing).
- `examples/research-sota/r6_fresnel_results.json` — machine-readable predictions.
- `docs/research/sota-2026-05-22/R6-fresnel-forward-model.md` — research note with the model, the demo headline numbers, what it gives each existing workspace module, R12's revision path with a basis, R10 range correction, honest scope.

## Headline numbers

**First Fresnel envelope (the "channel of maximum sensitivity"):**

| Link | 2.4 GHz @ midpoint | 5 GHz @ midpoint |
|---|---:|---:|
| 2 m | 25 cm | 17 cm |
| 5 m | **40 cm** | 27 cm |
| 10 m | 56 cm | 39 cm |

A typical bedroom 5 m WiFi link has a ~40 cm wide ellipsoid where human occupancy dominates the CSI. Outside that, you're picking up only diffracted edge contributions.

**Per-subcarrier phase predictions** confirm what R5 measured experimentally: inside zone-1, phase spread across 20 MHz is < 0.5° (band-flat); outside zone-1, spread grows to 15° (band-dispersed). R5's band-spread top-subcarriers are now physically explained, not just measured.

## Why this matters for the research loop

Three earlier threads were forced to **bootstrap from data** because no forward model existed:

- **R7** (mincut adversarial) — could only detect inconsistency, not predict expected. With R6, "physically inconsistent" has a precise definition: residual ≥ noise floor on all links simultaneously.
- **R10** (foliage range) — used FSPL + ITU foliage but ignored Fresnel-zone obstruction. R6 says the 100 m sparse-foliage range should be retracted to ~70 m (zone obstruction adds ~30% discount).
- **R12** (eigenshift, negative result) — failed because SVD spectrum loses spatial structure. R6's forward operator is the basis that R12's PABS revision needs.

## Coordination

Tick-8 via `ticks/tick-8.md`. No PROGRESS.md edit. Branch `research/sota-r6-fresnel-forward`.

## Remaining threads

R1 (ToA multistatic), R2 (room field model — partly subsumed by R6+R12 path), R3 (cross-room re-ID), R4 (federated learning), R11 (through-bulkhead maritime), R13 (contactless BP), R15 (RF biometric across rooms).

~6.6h to cron stop (12:00 UTC).
